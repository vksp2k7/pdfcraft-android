//! Display-only updates for layered signature appearances.
//!
//! This module never changes the signed source bytes; it creates an in-memory display copy so a
//! trusted signature can replace the baked question-mark caption with a green checkmark.

use std::collections::HashSet;
use pdfcraft_cos::{Dict, Document, ObjRef, Object, Stream};
use super::{fmt, SignatureInfo, Status};
use super::listing::{nums, sig_fields};

fn unhex(s: &[u8]) -> Option<Vec<u8>> {
    let digits: Vec<u8> = s.iter().copied().filter(|c| !c.is_ascii_whitespace()).collect();
    let val = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
    digits.chunks(2).map(|p| Some(val(*p.first()?)? << 4 | p.get(1).map_or(Some(0), |c| val(*c))?)).collect()
}

/// A copy of `doc` in which every validated (`Status::Valid`), visible signature widget that
/// carries a layered unverified appearance (a frame Form XObject invoking a standalone unverified
/// caption layer, a separate signer-details text layer, and a vector question-mark/status glyph
/// layer) has its caption updated to `"Signature valid"` and its glyph replaced by a green
/// checkmark drawn behind the text layers. Returns `None` when no widget qualifies.
pub fn with_validated_appearances(doc: &Document, signatures: &[SignatureInfo]) -> Option<Document> {
    if !signatures.iter().any(|s| s.signed && s.visible && s.status == Status::Valid) {
        return None;
    }
    let mut copy: Option<Document> = None;
    for f in sig_fields(doc) {
        if !signatures.iter().any(|s| s.signed && s.visible && s.status == Status::Valid && s.field == f.name) {
            continue;
        }
        for (widget_ref, wd) in &f.widgets {
            let Some(widget_ref) = *widget_ref else { continue };
            let Some(patch) = plan_validated_widget(doc, widget_ref, wd) else { continue };
            apply_validated_widget(copy.get_or_insert_with(|| doc.clone()), patch);
        }
    }
    copy
}

struct ValidatedWidgetPatch {
    widget_ref: ObjRef,
    widget_dict: Dict,
    ap_ref: Option<ObjRef>,
    ap_dict: Dict,
    frame_dict: Dict,
    frame_xobjects: Dict,
    frame_ops: Vec<u8>,
    glyph_ref: ObjRef,
    glyph_dict: Dict,
    glyph_ops: Vec<u8>,
    caption_ref: ObjRef,
    caption_dict: Dict,
    caption_ops: Vec<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LayerRole {
    Caption,
    Words,
    Glyph,
    Other,
}

#[derive(Clone)]
struct DoCall {
    name: Vec<u8>,
    matrix: Option<[f64; 6]>,
}

struct FormScan {
    do_calls: Vec<DoCall>,
    has_text_ops: bool,
    text_lines: Vec<String>,
    path_points: Vec<(f64, f64)>,
    has_curves: bool,
    has_re: bool,
    has_inline_image: bool,
    has_other_ops: bool,
    has_unwrapped_do: bool,
}

/// Plans a widget-local appearance rewrite for a supported layered valid signature.
fn plan_validated_widget(doc: &Document, widget_ref: ObjRef, wd: &Dict) -> Option<ValidatedWidgetPatch> {
    let ap = doc.resolve(wd.get(b"AP")?).as_dict().cloned()?;
    let ap_ref = wd.get(b"AP").and_then(Object::as_ref);
    let ap_n = ap.reference(b"N")?;
    let (_frame_ref, frame_stream, frame_bbox, xobjects, frame_scan) = find_frame_form(doc, ap_n)?;
    let mut roles: Vec<(DoCall, ObjRef, LayerRole)> = Vec::with_capacity(frame_scan.do_calls.len());
    for call in &frame_scan.do_calls {
        let layer_ref = xobjects.reference(&call.name)?;
        let Object::Stream(layer_stream) = &*doc.get(layer_ref) else { return None };
        if layer_stream.dict.name(b"Subtype") != Some(b"Form") {
            return None;
        }
        let layer_bytes = layer_stream.decoded().ok()?;
        let lscan = scan_form_stream(&layer_bytes);
        if lscan.has_inline_image {
            return None;
        }
        // A layer that invokes an Image XObject is raster artwork, never a vector status glyph.
        if !lscan.do_calls.is_empty() {
            roles.push((call.clone(), layer_ref, LayerRole::Other));
            continue;
        }
        let role = if is_unverified_caption(&lscan.text_lines) {
            LayerRole::Caption
        } else if is_question_mark_text(&lscan.text_lines) {
            LayerRole::Glyph
        } else if lscan.has_text_ops {
            LayerRole::Words
        } else if !lscan.path_points.is_empty() {
            let lbbox = form_bbox(doc, &layer_stream.dict)?;
            if lscan.has_curves || !is_perimeter_border(&lscan.path_points, lbbox) { LayerRole::Glyph } else { LayerRole::Other }
        } else {
            LayerRole::Other
        };
        roles.push((call.clone(), layer_ref, role));
    }
    let mut caption_refs: Vec<ObjRef> = roles.iter().filter(|(_, _, r)| *r == LayerRole::Caption).map(|(_, o, _)| *o).collect();
    caption_refs.sort_unstable_by_key(|r| (r.num, r.generation));
    caption_refs.dedup();
    let [caption_ref] = caption_refs.as_slice() else { return None };

    let has_words = roles.iter().any(|(_, _, r)| *r == LayerRole::Words);
    if !has_words {
        return None;
    }

    let mut glyph_refs: Vec<ObjRef> = roles.iter().filter(|(_, _, r)| *r == LayerRole::Glyph).map(|(_, o, _)| *o).collect();
    glyph_refs.sort_unstable_by_key(|r| (r.num, r.generation));
    glyph_refs.dedup();
    let [glyph_ref] = glyph_refs.as_slice() else { return None };

    if caption_ref == glyph_ref
        || roles.iter().any(|(_, o, r)| (*o == *caption_ref && *r != LayerRole::Caption) || (*o == *glyph_ref && *r != LayerRole::Glyph))
    {
        return None;
    }

    let Object::Stream(glyph_stream) = &*doc.get(*glyph_ref) else { return None };
    let mut glyph_dict = glyph_stream.dict.clone();
    glyph_dict.remove(b"Filter");
    glyph_dict.remove(b"DecodeParms");
    glyph_dict.remove(b"Matrix");
    glyph_dict.set(b"BBox".to_vec(), Object::Array(frame_bbox.iter().map(|v| Object::Real(*v)).collect()));
    let glyph_ops = valid_glyph_ops(frame_bbox);

    let Object::Stream(cap_stream) = &*doc.get(*caption_ref) else { return None };
    let cap_bbox = form_bbox(doc, &cap_stream.dict)?;
    let mut caption_dict = cap_stream.dict.clone();
    caption_dict.remove(b"Filter");
    caption_dict.remove(b"DecodeParms");
    let mut cap_res = caption_dict.get(b"Resources").map(|r| doc.resolve(r)).and_then(|r| r.as_dict().cloned()).unwrap_or_default();
    let mut cap_fonts = cap_res.get(b"Font").map(|f| doc.resolve(f)).and_then(|f| f.as_dict().cloned()).unwrap_or_default();
    let mut helv = Dict::new();
    helv.set(b"Type".to_vec(), Object::name("Font"));
    helv.set(b"Subtype".to_vec(), Object::name("Type1"));
    helv.set(b"BaseFont".to_vec(), Object::name("Helvetica"));
    helv.set(b"Encoding".to_vec(), Object::name("WinAnsiEncoding"));
    cap_fonts.set(b"F1".to_vec(), Object::Dict(helv));
    cap_res.set(b"Font".to_vec(), Object::Dict(cap_fonts));
    caption_dict.set(b"Resources".to_vec(), Object::Dict(cap_res));
    let caption_ops = valid_caption_ops(cap_bbox, frame_bbox);

    // Draw the glyph layer before the text layers (Words and Caption) so the green tick sits
    // behind the signer text and caption instead of obscuring them.
    let mut ordered = roles;
    if let (Some(first_text), Some(glyph_pos)) = (
        ordered.iter().position(|(_, _, r)| matches!(r, LayerRole::Words | LayerRole::Caption)),
        ordered.iter().position(|(_, _, r)| *r == LayerRole::Glyph),
    ) && glyph_pos > first_text
    {
        let item = ordered.remove(glyph_pos);
        ordered.insert(first_text, item);
    }
    let mut frame_ops = String::new();
    for (call, _, role) in &ordered {
        let name = String::from_utf8_lossy(&call.name);
        match (role, call.matrix) {
            (LayerRole::Glyph, _) | (_, None) => {
                frame_ops.push_str(&format!("q /{name} Do Q\n"));
            }
            (_, Some([a, b, c, d, e, f])) => {
                frame_ops.push_str(&format!("q {} {} {} {} {} {} cm /{name} Do Q\n", fmt(a), fmt(b), fmt(c), fmt(d), fmt(e), fmt(f)));
            }
        }
    }
    let mut frame_dict = frame_stream.dict.clone();
    frame_dict.remove(b"Filter");
    frame_dict.remove(b"DecodeParms");

    Some(ValidatedWidgetPatch {
        widget_ref,
        widget_dict: wd.clone(),
        ap_ref,
        ap_dict: ap,
        frame_dict,
        frame_xobjects: xobjects,
        frame_ops: frame_ops.into_bytes(),
        glyph_ref: *glyph_ref,
        glyph_dict,
        glyph_ops,
        caption_ref: *caption_ref,
        caption_dict,
        caption_ops,
    })
}

/// Applies a validated appearance patch using cloned widget-local PDF objects.
fn apply_validated_widget(doc: &mut Document, patch: ValidatedWidgetPatch) {
    let glyph_ref = doc.add(Object::Stream(Stream::from_raw(patch.glyph_dict, patch.glyph_ops)));
    let caption_ref = doc.add(Object::Stream(Stream::from_raw(patch.caption_dict, patch.caption_ops)));
    let frame_xobjects = patch
        .frame_xobjects
        .iter()
        .map(|(name, object)| {
            let object = match object.as_ref() {
                Some(reference) if reference == patch.glyph_ref => Object::Ref(glyph_ref),
                Some(reference) if reference == patch.caption_ref => Object::Ref(caption_ref),
                _ => object.clone(),
            };
            (name.clone(), object)
        })
        .collect::<Dict>();
    let mut frame_dict = patch.frame_dict;
    let mut resources = frame_dict.get(b"Resources").map(|r| doc.resolve(r)).and_then(|r| r.as_dict().cloned()).unwrap_or_default();
    resources.set(b"XObject".to_vec(), Object::Dict(frame_xobjects));
    let resources_ref = doc.add(Object::Dict(resources));
    frame_dict.set(b"Resources".to_vec(), Object::Ref(resources_ref));
    let frame_ref = doc.add(Object::Stream(Stream::from_raw(frame_dict, patch.frame_ops)));
    let mut ap_dict = patch.ap_dict;
    ap_dict.set(b"N".to_vec(), Object::Ref(frame_ref));
    let mut widget_dict = patch.widget_dict;
    if patch.ap_ref.is_some() {
        let ap_ref = doc.add(Object::Dict(ap_dict));
        widget_dict.set(b"AP".to_vec(), Object::Ref(ap_ref));
    } else {
        widget_dict.set(b"AP".to_vec(), Object::Dict(ap_dict));
    }
    doc.set(patch.widget_ref, Object::Dict(widget_dict));
}

/// Finds the safe layered frame form beneath an appearance wrapper.
fn find_frame_form(doc: &Document, start: ObjRef) -> Option<(ObjRef, Stream, [f64; 4], Dict, FormScan)> {
    let mut curr = start;
    let mut seen = HashSet::new();
    for _ in 0..4 {
        if !seen.insert(curr) {
            return None;
        }
        let Object::Stream(s) = &*doc.get(curr) else { return None };
        if s.dict.name(b"Subtype") != Some(b"Form") {
            return None;
        }
        let bytes = s.decoded().ok()?;
        let scan = scan_form_stream(&bytes);
        if scan.has_inline_image {
            return None;
        }
        let res = s.dict.get(b"Resources").map(|r| doc.resolve(r)).and_then(|r| r.as_dict().cloned())?;
        let xobjects = res.get(b"XObject").map(|x| doc.resolve(x)).and_then(|x| x.as_dict().cloned())?;
        if scan.do_calls.len() == 1 && !scan.has_other_ops && !scan.has_unwrapped_do {
            let call = scan.do_calls.first()?;
            let wrapper_transformed =
                nums(doc, &s.dict, b"Matrix").and_then(|m| <[f64; 6]>::try_from(m.as_slice()).ok()).is_some_and(|m| !is_identity_matrix(m));
            if wrapper_transformed || call.matrix.is_some() {
                return None;
            }
            curr = xobjects.reference(&call.name)?;
            continue;
        }
        if scan.do_calls.len() >= 2 && !scan.has_other_ops && !scan.has_unwrapped_do {
            let bbox = form_bbox(doc, &s.dict)?;
            return Some((curr, s.clone(), bbox, xobjects, scan));
        }
        return None;
    }
    None
}

/// Reads and normalizes a finite, non-empty Form XObject bounding box.
fn form_bbox(doc: &Document, d: &Dict) -> Option<[f64; 4]> {
    let b = nums(doc, d, b"BBox")?;
    let [x0, y0, x1, y1] = <[f64; 4]>::try_from(b.as_slice()).ok()?;
    if ![x0, y0, x1, y1].iter().all(|v| v.is_finite()) {
        return None;
    }
    let (min_x, max_x) = (x0.min(x1), x0.max(x1));
    let (min_y, max_y) = (y0.min(y1), y0.max(y1));
    (max_x - min_x > 0.5 && max_y - min_y > 0.5).then_some([min_x, min_y, max_x, max_y])
}

/// Reports whether extracted layer text is a known unverified-signature caption.
fn is_unverified_caption(lines: &[String]) -> bool {
    if lines.is_empty() {
        return false;
    }
    let joined: String = lines.iter().flat_map(|l| l.split_whitespace()).collect::<Vec<_>>().join(" ").to_ascii_lowercase();
    matches!(joined.as_str(), "signature not verified" | "validity unknown" | "signature validity is unknown")
}

/// Reports whether extracted layer text is the standalone question-mark glyph.
fn is_question_mark_text(lines: &[String]) -> bool {
    let joined: String = lines.iter().flat_map(|l| l.split_whitespace()).collect::<Vec<_>>().join(" ");
    joined == "?"
}

/// Reports whether path points form a border around most of a layer’s bounding box.
fn is_perimeter_border(points: &[(f64, f64)], bbox: [f64; 4]) -> bool {
    if points.len() < 4 {
        return false;
    }
    let [bx0, by0, bx1, by1] = bbox;
    let (bw, bh) = (bx1 - bx0, by1 - by0);
    if bw <= 0.0 || bh <= 0.0 {
        return false;
    }
    let min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let max_x = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let min_y = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_y = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    if max_x - min_x < 0.85 * bw || max_y - min_y < 0.85 * bh {
        return false;
    }
    let margin = 0.12 * bw.min(bh);
    points.iter().all(|&(x, y)| {
        let dx = (x - bx0).abs().min((bx1 - x).abs());
        let dy = (y - by0).abs().min((by1 - y).abs());
        dx.min(dy) <= margin
    })
}

/// Builds PDF operators for the green checkmark used in a validated display copy.
fn valid_glyph_ops(bbox: [f64; 4]) -> Vec<u8> {
    let [bx0, by0, bx1, by1] = bbox;
    let (bw, bh) = (bx1 - bx0, by1 - by0);
    let s = bw.min(bh);
    let ox = bx0 + (bw - s) * 0.5;
    let oy = by0 + (bh - s) * 0.5;
    let d = s * 0.03;
    let lw = (s * 0.012).max(0.5);
    const PTS: [(f64, f64); 6] = [(0.15, 0.42), (0.23, 0.53), (0.40, 0.34), (0.74, 0.86), (0.85, 0.78), (0.42, 0.16)];
    let mut out = String::from("q\n0 g\n");
    for (i, &(px, py)) in PTS.iter().enumerate() {
        let x = ox + px * s + d;
        let y = oy + py * s - d;
        out.push_str(&format!("{} {} {}\n", fmt(x), fmt(y), if i == 0 { "m" } else { "l" }));
    }
    out.push_str(&format!("h f\n0 G\n0 0.68 0.15 rg\n{} w\n0 j\n", fmt(lw)));
    for (i, &(px, py)) in PTS.iter().enumerate() {
        let x = ox + px * s;
        let y = oy + py * s;
        out.push_str(&format!("{} {} {}\n", fmt(x), fmt(y), if i == 0 { "m" } else { "l" }));
    }
    out.push_str("h B\nQ\n");
    out.into_bytes()
}

/// Builds PDF text operators for the validated signature caption.
fn valid_caption_ops(cap_bbox: [f64; 4], frame_bbox: [f64; 4]) -> Vec<u8> {
    let [bx0, by0, bx1, by1] = cap_bbox;
    let (bw, bh) = ((bx1 - bx0).max(1.0), (by1 - by0).max(1.0));
    let frame_bh = (frame_bbox[3] - frame_bbox[1]).abs();
    // When a caption layer shares the full widget BBox instead of a dedicated top strip,
    // place the caption in the top 30% strip where standard layered captions sit.
    let (strip_y, strip_h) = if frame_bh > 0.0 && (bh - frame_bh).abs() < 1.0 && bh > bw * 0.45 { (by0 + bh * 0.70, bh * 0.30) } else { (by0, bh) };
    let label = "Signature valid";
    let pad = (bw * 0.015).clamp(0.5, 4.0);
    let unit_w = pdfcraft_fonts::helvetica_width(label, 1.0).max(1.0);
    let size = ((bw - 2.0 * pad).max(1.0) / unit_w).min(strip_h * 0.85).max(1.0);
    let x = bx0 + pad;
    let y = strip_y + ((strip_h - size * 0.72).max(0.0)) * 0.5;
    format!("BT\n0 g\n/F1 {} Tf\n{} {} Td\n({label}) Tj\nET\n", fmt(size), fmt(x), fmt(y)).into_bytes()
}

#[derive(Clone)]
enum StreamToken {
    Name(Vec<u8>),
    Number(f64),
    Str(Vec<u8>),
    Array(Vec<StreamToken>),
    Op(Vec<u8>),
}

fn is_pdf_ws(b: u8) -> bool {
    matches!(b, 0x00 | 0x09 | 0x0A | 0x0C | 0x0D | 0x20)
}

fn is_pdf_delim(b: u8) -> bool {
    matches!(b, b'/' | b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'%')
}

/// Parses the next token from a PDF content stream, including nested arrays.
fn next_stream_token(bytes: &[u8], pos: &mut usize, depth: usize) -> Option<StreamToken> {
    if depth > 6 {
        return None;
    }
    while let Some(&b) = bytes.get(*pos) {
        if is_pdf_ws(b) {
            *pos += 1;
        } else if b == b'%' {
            *pos += 1;
            while let Some(&c) = bytes.get(*pos) {
                *pos += 1;
                if c == b'\n' || c == b'\r' {
                    break;
                }
            }
        } else {
            break;
        }
    }
    let &first = bytes.get(*pos)?;
    match first {
        b'/' => {
            *pos += 1;
            let mut name = Vec::new();
            while let Some(&b) = bytes.get(*pos) {
                if is_pdf_ws(b) || is_pdf_delim(b) {
                    break;
                }
                if b == b'#'
                    && let (Some(&h1), Some(&h2)) = (bytes.get(*pos + 1), bytes.get(*pos + 2))
                    && let (Some(d1), Some(d2)) = ((h1 as char).to_digit(16), (h2 as char).to_digit(16))
                {
                    name.push(((d1 << 4) | d2) as u8);
                    *pos += 3;
                    continue;
                }
                name.push(b);
                *pos += 1;
            }
            Some(StreamToken::Name(name))
        }
        b'(' => {
            *pos += 1;
            let mut out = Vec::new();
            let mut paren = 1usize;
            while let Some(&b) = bytes.get(*pos) {
                *pos += 1;
                match b {
                    b'(' => {
                        paren += 1;
                        out.push(b'(');
                    }
                    b')' => {
                        paren -= 1;
                        if paren == 0 {
                            break;
                        }
                        out.push(b')');
                    }
                    b'\\' => {
                        let Some(&esc) = bytes.get(*pos) else { break };
                        *pos += 1;
                        match esc {
                            b'n' => out.push(b'\n'),
                            b'r' => out.push(b'\r'),
                            b't' => out.push(b'\t'),
                            b'b' => out.push(0x08),
                            b'f' => out.push(0x0C),
                            b'(' => out.push(b'('),
                            b')' => out.push(b')'),
                            b'\\' => out.push(b'\\'),
                            b'\r' => {
                                if bytes.get(*pos) == Some(&b'\n') {
                                    *pos += 1;
                                }
                            }
                            b'\n' => {}
                            b'0'..=b'7' => {
                                let mut oct = u32::from(esc - b'0');
                                for _ in 0..2 {
                                    if let Some(&d @ b'0'..=b'7') = bytes.get(*pos) {
                                        oct = (oct << 3) | u32::from(d - b'0');
                                        *pos += 1;
                                    } else {
                                        break;
                                    }
                                }
                                out.push((oct & 0xFF) as u8);
                            }
                            other => out.push(other),
                        }
                    }
                    other => out.push(other),
                }
            }
            Some(StreamToken::Str(out))
        }
        b'<' => {
            if bytes.get(*pos + 1) == Some(&b'<') {
                *pos += 2;
                Some(StreamToken::Op(b"<<".to_vec()))
            } else {
                *pos += 1;
                let mut hex = Vec::new();
                while let Some(&b) = bytes.get(*pos) {
                    *pos += 1;
                    if b == b'>' {
                        break;
                    }
                    hex.push(b);
                }
                Some(StreamToken::Str(unhex(&hex).unwrap_or_default()))
            }
        }
        b'>' => {
            if bytes.get(*pos + 1) == Some(&b'>') {
                *pos += 2;
                Some(StreamToken::Op(b">>".to_vec()))
            } else {
                *pos += 1;
                Some(StreamToken::Op(b">".to_vec()))
            }
        }
        b'[' => {
            *pos += 1;
            let mut items = Vec::new();
            while let Some(&b) = bytes.get(*pos) {
                if is_pdf_ws(b) {
                    *pos += 1;
                    continue;
                }
                if b == b']' {
                    *pos += 1;
                    break;
                }
                let before = *pos;
                if let Some(tok) = next_stream_token(bytes, pos, depth + 1) {
                    items.push(tok);
                }
                if *pos == before {
                    *pos += 1;
                }
            }
            Some(StreamToken::Array(items))
        }
        b']' => {
            *pos += 1;
            Some(StreamToken::Op(b"]".to_vec()))
        }
        _ => {
            let mut raw = Vec::new();
            while let Some(&b) = bytes.get(*pos) {
                if is_pdf_ws(b) || is_pdf_delim(b) {
                    break;
                }
                raw.push(b);
                *pos += 1;
            }
            if let Ok(s) = std::str::from_utf8(&raw)
                && let Ok(n) = s.parse::<f64>()
                && n.is_finite()
            {
                return Some(StreamToken::Number(n));
            }
            Some(StreamToken::Op(raw))
        }
    }
}

/// Decodes PDFDocEncoding bytes into text used while classifying appearance layers.
fn decode_pdf_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|b| pdfcraft_cos::pdfdoc_char(*b)).collect()
}

/// Multiplies two PDF affine transformation matrices in content-stream order.
fn concat_matrix(a: [f64; 6], b: [f64; 6]) -> [f64; 6] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}

/// Reports whether a PDF affine matrix is effectively the identity transform.
fn is_identity_matrix(m: [f64; 6]) -> bool {
    (m[0] - 1.0).abs() < 1e-4 && m[1].abs() < 1e-4 && m[2].abs() < 1e-4 && (m[3] - 1.0).abs() < 1e-4 && m[4].abs() < 1e-4 && m[5].abs() < 1e-4
}

/// Scans a Form XObject stream for text, paths, nested layers, and unsafe operators.
fn scan_form_stream(bytes: &[u8]) -> FormScan {
    let mut scan = FormScan {
        do_calls: Vec::new(),
        has_text_ops: false,
        text_lines: Vec::new(),
        path_points: Vec::new(),
        has_curves: false,
        has_re: false,
        has_inline_image: false,
        has_other_ops: false,
        has_unwrapped_do: false,
    };
    let mut pos = 0usize;
    let mut args: Vec<StreamToken> = Vec::new();
    let mut ctm_stack: Vec<[f64; 6]> = vec![[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
    let mut current_line = String::new();
    let mut last_tm_y: Option<f64> = None;
    let flush_line = |line: &mut String, lines: &mut Vec<String>| {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            lines.push(trimmed.to_string());
        }
        line.clear();
    };
    while pos < bytes.len() {
        let before = pos;
        let Some(tok) = next_stream_token(bytes, &mut pos, 0) else { break };
        if pos == before {
            pos += 1;
        }
        match tok {
            StreamToken::Op(op) => {
                if !matches!(op.as_slice(), b"q" | b"Q" | b"cm" | b"Do") {
                    scan.has_other_ops = true;
                }
                match op.as_slice() {
                    b"BI" => {
                        scan.has_inline_image = true;
                        break;
                    }
                    b"q" => {
                        if ctm_stack.len() < 32
                            && let Some(&top) = ctm_stack.last()
                        {
                            ctm_stack.push(top);
                        }
                    }
                    b"Q" => {
                        if ctm_stack.len() > 1 {
                            ctm_stack.pop();
                        } else if let Some(top) = ctm_stack.first_mut() {
                            *top = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
                        }
                    }
                    b"cm" => {
                        if let Some(nums) = last_n_numbers(&args, 6)
                            && let Ok(m) = <[f64; 6]>::try_from(nums.as_slice())
                            && let Some(top) = ctm_stack.last_mut()
                        {
                            *top = concat_matrix(m, *top);
                        }
                    }
                    b"Do" => {
                        if ctm_stack.len() == 1 {
                            scan.has_unwrapped_do = true;
                        }
                        if let Some(StreamToken::Name(name)) = args.last() {
                            let ctm = ctm_stack.last().copied().unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
                            scan.do_calls.push(DoCall { name: name.clone(), matrix: (!is_identity_matrix(ctm)).then_some(ctm) });
                        }
                    }
                    b"BT" => {
                        scan.has_text_ops = true;
                        last_tm_y = None;
                    }
                    b"ET" => {
                        scan.has_text_ops = true;
                        flush_line(&mut current_line, &mut scan.text_lines);
                        last_tm_y = None;
                    }
                    b"Td" | b"TD" => {
                        if let Some(nums) = last_n_numbers(&args, 2)
                            && let [_, ty] = nums.as_slice()
                            && ty.abs() > 0.01
                        {
                            flush_line(&mut current_line, &mut scan.text_lines);
                        }
                    }
                    b"T*" => {
                        flush_line(&mut current_line, &mut scan.text_lines);
                    }
                    b"Tm" => {
                        if let Some(nums) = last_n_numbers(&args, 6)
                            && let [.., ty] = nums.as_slice()
                        {
                            if last_tm_y.is_some_and(|prev| (*ty - prev).abs() > 0.1) {
                                flush_line(&mut current_line, &mut scan.text_lines);
                            }
                            last_tm_y = Some(*ty);
                        }
                    }
                    b"Tj" => {
                        scan.has_text_ops = true;
                        if let Some(StreamToken::Str(s)) = args.last() {
                            current_line.push_str(&decode_pdf_bytes(s));
                        }
                    }
                    b"'" | b"\"" => {
                        scan.has_text_ops = true;
                        flush_line(&mut current_line, &mut scan.text_lines);
                        if let Some(StreamToken::Str(s)) = args.last() {
                            current_line.push_str(&decode_pdf_bytes(s));
                        }
                    }
                    b"TJ" => {
                        scan.has_text_ops = true;
                        if let Some(StreamToken::Array(items)) = args.last() {
                            for item in items {
                                match item {
                                    StreamToken::Str(s) => current_line.push_str(&decode_pdf_bytes(s)),
                                    StreamToken::Number(n) if *n <= -120.0 && !current_line.ends_with(' ') => current_line.push(' '),
                                    _ => {}
                                }
                            }
                        }
                    }
                    b"m" | b"l" => {
                        if let Some(nums) = last_n_numbers(&args, 2)
                            && let [x, y] = nums.as_slice()
                        {
                            scan.path_points.push((*x, *y));
                        }
                    }
                    b"v" | b"y" => {
                        scan.has_curves = true;
                        if let Some(nums) = last_n_numbers(&args, 4)
                            && let [x1, y1, x2, y2] = nums.as_slice()
                        {
                            scan.path_points.push((*x1, *y1));
                            scan.path_points.push((*x2, *y2));
                        }
                    }
                    b"c" => {
                        scan.has_curves = true;
                        if let Some(nums) = last_n_numbers(&args, 6)
                            && let [x1, y1, x2, y2, x3, y3] = nums.as_slice()
                        {
                            scan.path_points.push((*x1, *y1));
                            scan.path_points.push((*x2, *y2));
                            scan.path_points.push((*x3, *y3));
                        }
                    }
                    b"re" if last_n_numbers(&args, 4).is_some() => {
                        scan.has_re = true;
                    }
                    _ => {}
                }
                args.clear();
            }
            other => {
                if args.len() >= 16 {
                    args.remove(0);
                }
                args.push(other);
            }
        }
    }
    flush_line(&mut current_line, &mut scan.text_lines);
    scan
}

fn last_n_numbers(args: &[StreamToken], n: usize) -> Option<Vec<f64>> {
    if args.len() < n {
        return None;
    }
    let slice = args.get(args.len() - n..)?;
    let mut out = Vec::with_capacity(n);
    for tok in slice {
        match tok {
            StreamToken::Number(v) => out.push(*v),
            _ => return None,
        }
    }
    Some(out)
}
