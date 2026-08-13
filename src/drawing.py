"""
drawing.py  --  STAGE 5 of the pipeline:  draw bbox + label onto the frame.

============================ THE BIG PICTURE ================================
Detection gives us NUMBERS (box coordinates + confidence). Numbers are hard
to eyeball. So we draw those boxes back ONTO the image, plus a little text
label, and show it in a window. This is purely for US, the humans, to confirm
"yes, it's finding the people correctly". The neural network doesn't need
this; our eyes do. This is called VISUALIZATION or an "annotated frame".

Everything here is done with OpenCV's simple drawing functions, which paint
directly onto the pixel array (they modify the image in place).

Reminder: OpenCV colours are (Blue, Green, Red), each 0-255. So:
  (0, 255, 0)   = pure green
  (0, 0, 255)   = pure red
  (255, 0, 0)   = pure blue
============================================================================
"""

import zlib

import cv2

# The colour we draw when there is NO track id yet (plain detection). Green.
BOX_COLOR = (0, 255, 0)          # BGR = green
TEXT_COLOR = (0, 0, 0)           # BGR = black (drawn on the coloured label patch)
BOX_THICKNESS = 3                # line thickness in pixels
FONT = cv2.FONT_HERSHEY_SIMPLEX  # a built-in OpenCV font
LABEL_FONT_SCALE = 1.0           # id/GID label size (was 0.5)
LABEL_THICKNESS = 2              # id/GID label stroke weight (was 1)

# A fixed palette of distinct, bright BGR colours. Each track id is mapped to
# one of these so the SAME person keeps the SAME colour frame to frame, which
# makes it easy to follow someone with your eyes.
# 20 colours, not 8 (plan #35). With 8, `id % 8` made ids 3 and 11 the same colour,
# and across several camera videos the check an operator actually performs is "did
# this person keep their colour" -- so a collision READS AS a false merge that never
# happened. Every entry below is distinguishable from the others in BGR at video
# scale; the ordering interleaves hues so that CONSECUTIVE ids (the common case,
# since gids are assigned in order) are always far apart.
UNRESOLVED_COLOR = (150, 150, 150)   # neutral grey, never an identity

_PALETTE = [
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 0, 255),      # red
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 165, 255),    # orange
    (128, 0, 255),    # pink
    (0, 128, 0),      # dark green
    (255, 128, 0),    # azure
    (0, 0, 128),      # maroon
    (128, 255, 255),  # pale yellow
    (203, 192, 255),  # light pink
    (0, 215, 255),    # gold
    (139, 61, 72),    # plum
    (170, 178, 32),   # olive
    (255, 191, 0),    # deep sky
    (147, 20, 255),   # deep pink
    (79, 79, 47),     # dark slate
    (212, 255, 127),  # aquamarine
]


def color_for_id(track_id):
    """
    Return a stable colour for a given track id. We use the id modulo the
    palette length, so id 0 and id 8 share a colour but consecutive ids differ.
    If there's no id (plain detection), fall back to green.
    """
    if track_id is None:
        return BOX_COLOR
    if not isinstance(track_id, int):
        track_id = zlib.crc32(str(track_id).encode("utf-8"))
    return _PALETTE[int(track_id) % len(_PALETTE)]


def draw_detections(frame, detections):
    """
    Draw every detection onto `frame` and return the annotated frame.

    NOTE: OpenCV draws IN PLACE (it edits the array you pass in). We return it
    too, just so the calling code reads naturally.
    """
    for det in detections:
        # A NEGATIVE reid_id is a PROVISIONAL live label: an online-mode track
        # that is still gathering evidence and hasn't resolved to a real global
        # id yet (see IdentityService._assign_online). Treat it as "not yet
        # identified" for both colour and label, so the overlay shows a neutral
        # "pending" marker instead of a scary "REID -1" that then flips.
        provisional = det.reid_id is not None and det.reid_id < 0
        # #32/#33: a tracklet reconcile could not resolve (suppressed below
        # min_tracklet_observations, or left over) arrives with NO identity at all.
        # It used to render as a bare "ID 47", which reads as the identity VANISHING
        # -- and worse, as a *different person* than the same body a second earlier.
        # Mark it as unresolved instead, and never print a number that could be
        # mistaken for an identity.
        unresolved = (det.reid_id is None and det.global_id is None
                      and getattr(det, "unresolved", True))

        # Colour by REID id when we have one, so the same person keeps the same
        # colour ACROSS cameras (that's the whole point of ReID). Fall back to
        # the per-camera track id, then green. Provisional tracks colour by
        # track id (stable while pending) rather than the throwaway negative id.
        ident = (det.reid_id if (det.reid_id is not None and not provisional)
                 else det.global_id if det.global_id is not None
                 else det.track_id)
        # Unresolved gets ONE fixed neutral grey, deliberately not from the palette:
        # a palette colour would make it look like a specific identity, and two
        # different unresolved people would look like the same one.
        color = UNRESOLVED_COLOR if unresolved else color_for_id(ident)

        # 1) The rectangle. cv2.rectangle needs the two opposite corners:
        #    top-left (x1, y1) and bottom-right (x2, y2).
        cv2.rectangle(
            frame,
            (det.x1, det.y1),   # top-left corner
            (det.x2, det.y2),   # bottom-right corner
            color,
            BOX_THICKNESS,
        )

        # 2) The text label. Prefer the cross-camera REID id, else the legacy
        #    global id, else just the track id ("ID 3  0.87"), else "person".
        #
        # The score shown next to a REID id is the IDENTITY match cosine
        # (det.reid_score), NOT det.confidence. det.confidence answers "is this a
        # person"; when the question on screen is "is this the SAME person", a
        # detection score is the wrong number and misleads by looking plausible.
        #
        # No score -> print none. Two cases legitimately have no identity score:
        # a freshly MINTED identity (it matched nothing, so there is nothing to
        # report -- and "0.00" would read as a terrible match rather than no
        # comparison), and the offline re-render in main.py::render_final_videos,
        # whose ids come from tracklet-prototype merges on a different score
        # scale. getattr because that path passes SimpleNamespace, not Detection.
        reid_score = getattr(det, "reid_score", None)
        # #34: the FINAL video used to show no identity confidence whatsoever,
        # because render_final_videos built its detections without reid_score. Two
        # numbers are meaningful after reconcile and both come from the merge, not
        # from YOLO:
        #   fit    -- this tracklet's prototype against its final cluster prototype
        #             ("how well does this piece match the person it was filed as")
        #   margin -- gap to the nearest OTHER cluster ("how close was the call")
        # A low margin with a high fit is the signature of a merge worth doubting.
        # Rendered as `REID 7 (0.94 / +0.21)`. These are RECONCILE-scale and are NOT
        # comparable to the live engine's scores -- different comparison entirely.
        if unresolved:
            # #32/#33: never print a number that could be read as an identity.
            # But "UNRESOLVED" alone does not say WHY, and the causes need
            # opposite fixes while looking identical on screen -- a tracklet
            # under min_tracklet_observations was never compared to anything,
            # while one whose best candidate missed the bar by 0.01 was.
            #
            # Every field is optional and defaults to None, so this degrades to
            # a bare "UNRESOLVED" until reconcile populates them. The score is
            # ALWAYS printed against its threshold or with a reason attached,
            # so it cannot be mistaken for an identity number.
            best_score = getattr(det, "reid_best_score", None)
            threshold = getattr(det, "reid_threshold", None)
            reason = getattr(det, "reid_reason", None)
            best_candidate = getattr(det, "reid_best_candidate", None)
            best_margin = getattr(det, "reid_best_margin", None)

            parts = ["UNRESOLVED"]
            if best_candidate is not None:
                parts.append(f"->{best_candidate}")
            if best_score is not None:
                parts.append(f"{best_score:.3f}<{threshold:.3f}"
                             if threshold is not None else f"{best_score:.3f}")
            if best_margin is not None:
                parts.append(f"d{best_margin:+.3f}")
            if reason:
                parts.append(reason)
            label = " ".join(parts)

        elif provisional:
            # Still deciding who this is -- show a pending marker, not the id.
            label = (f"REID ...  ID{det.track_id}"
                     if det.track_id is not None else "REID ...")
        elif det.reid_id is not None:
            # The number is the score that LINKED this tracklet to another one,
            # against the bar it cleared -- NOT a fit against its own cluster
            # prototype, which contained itself and read 1.000 for anyone who
            # linked to nothing.
            link = getattr(det, "reid_link_score", None)
            bar = getattr(det, "reid_link_bar", None)
            partner = getattr(det, "reid_link_with", None)
            linked = getattr(det, "reid_linked", True)

            if link is None:
                # LIVE path: reconcile has not run, so fall back to the live
                # engine's own match cosine. Different comparison, different
                # scale -- parenthesised so it never reads as a reconcile score.
                head = f"REID {det.reid_id}"
                conf = f" ({reid_score:.2f})" if reid_score is not None else ""
            elif linked:
                head = f"REID {det.reid_id}"
                conf = (f" {link:.3f}>={bar:.3f}" if bar is not None
                        else f" {link:.3f}")
            else:
                # Kept its id and its colour -- a singleton with 200 observations
                # is still a person you can follow by eye. The "?" says only the
                # LINK is missing, and the score says how close it got.
                head = f"REID {det.reid_id}?"
                conf = (f" {link:.3f}<{bar:.3f}" if bar is not None
                        else f" {link:.3f}")
                if partner is not None:
                    conf += f" ->{partner}"

            label = (f"{head}{conf} ID{det.track_id}"
                     if det.track_id is not None else f"{head}{conf}")
        elif det.global_id is not None:
            label = (f"GID {det.global_id}  ID{det.track_id}"
                     if det.track_id is not None else f"GID {det.global_id}")
        elif det.track_id is not None:
            label = f"ID {det.track_id}  {det.confidence:.2f}"
        else:
            label = f"person {det.confidence:.2f}"

        # Measure how big that text will be, so we can draw a filled
        # background patch behind it -- otherwise the text can be unreadable.
        (text_w, text_h), baseline = cv2.getTextSize(
            label, FONT, LABEL_FONT_SCALE, LABEL_THICKNESS)

        # #36: the label used to be drawn unconditionally ABOVE the box, so for
        # anyone entering at the top of the frame (y1 < ~30) it landed at a negative
        # y and was clipped away entirely -- that person had NO visible id at the
        # exact moment identity is most in doubt. Flip it inside the box instead,
        # and clamp x so a box at the right edge keeps its text on screen.
        patch_h = text_h + baseline + 6
        frame_h, frame_w = frame.shape[:2]
        if det.y1 - patch_h >= 0:
            top = det.y1 - patch_h
        else:
            top = min(det.y1, max(0, frame_h - patch_h))      # inside the box
        left = min(det.x1, max(0, frame_w - (text_w + 4)))
        cv2.rectangle(
            frame,
            (left, top),
            (left + text_w + 4, top + patch_h),
            color,
            thickness=-1,   # -1 means "fill the rectangle solid"
        )
        cv2.putText(
            frame,
            label,
            (left + 2, top + text_h + 3),   # bottom-left anchor of the text
            FONT,
            LABEL_FONT_SCALE,
            TEXT_COLOR,
            LABEL_THICKNESS,
            cv2.LINE_AA,     # anti-aliased = smoother-looking text
        )

    return frame


def draw_hud(frame, person_count, fps=None):
    """
    Draw a small "heads-up display" in the top-left: how many people are in
    view, and (optionally) how fast we're processing, in frames per second.
    Handy for confirming the pipeline is keeping up in real time.
    """
    text = f"People: {person_count}"
    if fps is not None:
        text += f"   FPS: {fps:.1f}"

    cv2.putText(frame, text, (10, 30), FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return frame
