// Copyright (c) 2026, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.
#![deny(missing_docs)]

//! Bridge between rav1e frames and the `av-scenechange` crate.
//!
//! `av-scenechange` is built on `v_frame` 0.5, while rav1e's frames (and its
//! public API) still use `v_frame` 0.3, so the two `Frame` types are
//! unrelated. Each lookahead frame is therefore copied once into a `v_frame`
//! 0.5 frame before scene detection runs on it.
//!
//! Only the luma plane is copied, because `av-scenechange` never reads
//! chroma. The copy keeps the source plane's padding, so motion search near
//! the frame edges sees the same edge-extended pixels it did before.

use crate::api::{EncoderConfig, InterConfig, SceneDetectionSpeed};
use crate::color::ChromaSampling;
use crate::frame::Frame;
use crate::util::{CastFromPrimitive, Pixel};
// `av-scenechange` enables the `padding_api` feature of this `v_frame`,
// which provides `Plane::data_mut`.
use av_scenechange::av_decoders::v_frame::{
  chroma::ChromaSubsampling, frame::Frame as ScFrame, frame::FrameBuilder,
  pixel::Pixel as ScPixel,
};
use av_scenechange::{Rational32, SceneChangeDetector, ScenecutResult};
use std::num::{NonZeroU8, NonZeroUsize};
use std::sync::Arc;

/// Runs `av-scenechange` keyframe detection on rav1e frames.
pub(crate) struct KeyframeDetector<T: Pixel> {
  inner: Inner<T>,
}

/// `v_frame` 0.5 ties the pixel type to the bit depth (`u8` is 8-bit only and
/// `u16` is high bit depth only), whereas rav1e also allows 8-bit content in
/// `u16` frames. The detector's pixel type is therefore picked from the
/// configured bit depth rather than from `T`.
enum Inner<T: Pixel> {
  U8(Detector<T, u8>),
  U16(Detector<T, u16>),
}

impl<T: Pixel> KeyframeDetector<T> {
  pub(crate) fn new(enc: &EncoderConfig) -> Self {
    let inner = if enc.bit_depth == 8 {
      Inner::U8(Detector::new(enc))
    } else {
      Inner::U16(Detector::new(enc))
    };
    Self { inner }
  }

  /// Keeps the intra costs computed during scene detection so they can be
  /// retrieved with [`Self::take_intra_costs`].
  pub(crate) fn enable_cache(&mut self) {
    match &mut self.inner {
      Inner::U8(d) => d.detector.enable_cache(),
      Inner::U16(d) => d.detector.enable_cache(),
    }
  }

  /// Runs keyframe detection on the next frame in the lookahead queue.
  ///
  /// See [`SceneChangeDetector::analyze_next_frame`] for the meaning of the
  /// arguments.
  pub(crate) fn analyze_next_frame(
    &mut self, frame_set: &[&Arc<Frame<T>>], input_frameno: usize,
    previous_keyframe: usize,
  ) -> (bool, Option<ScenecutResult>) {
    match &mut self.inner {
      Inner::U8(d) => {
        d.analyze_next_frame(frame_set, input_frameno, previous_keyframe)
      }
      Inner::U16(d) => {
        d.analyze_next_frame(frame_set, input_frameno, previous_keyframe)
      }
    }
  }

  /// Removes and returns the cached intra costs for `input_frameno`, if
  /// caching is enabled and the frame has been analyzed.
  pub(crate) fn take_intra_costs(
    &mut self, input_frameno: usize,
  ) -> Option<Box<[u32]>> {
    let intra_costs = match &mut self.inner {
      Inner::U8(d) => d.detector.intra_costs.as_mut(),
      Inner::U16(d) => d.detector.intra_costs.as_mut(),
    };
    intra_costs?.remove(&input_frameno)
  }
}

struct Detector<T: Pixel, P: ScPixel> {
  detector: SceneChangeDetector<P>,
  bit_depth: NonZeroU8,
  /// Luma copies of the frames in the current lookahead window.
  frames: Vec<CachedFrame<T, P>>,
}

struct CachedFrame<T: Pixel, P: ScPixel> {
  /// The frame `luma` was copied from. Holding it keeps its address from
  /// being reused while the entry is cached.
  source: Arc<Frame<T>>,
  luma: Arc<ScFrame<P>>,
}

impl<T: Pixel, P: ScPixel + CastFromPrimitive<T>> Detector<T, P> {
  fn new(enc: &EncoderConfig) -> Self {
    let lookahead_distance =
      InterConfig::new(enc).keyframe_lookahead_distance() as usize;
    let detector = SceneChangeDetector::new(
      (enc.width, enc.height),
      enc.bit_depth,
      Rational32::new(enc.time_base.den as i32, enc.time_base.num as i32),
      match enc.chroma_sampling {
        ChromaSampling::Cs420 => ChromaSubsampling::Yuv420,
        ChromaSampling::Cs422 => ChromaSubsampling::Yuv422,
        ChromaSampling::Cs444 => ChromaSubsampling::Yuv444,
        ChromaSampling::Cs400 => ChromaSubsampling::Monochrome,
      },
      lookahead_distance,
      match enc.speed_settings.scene_detection_mode {
        SceneDetectionSpeed::Fast => av_scenechange::SceneDetectionSpeed::Fast,
        SceneDetectionSpeed::Standard => {
          av_scenechange::SceneDetectionSpeed::Standard
        }
        SceneDetectionSpeed::None => av_scenechange::SceneDetectionSpeed::None,
      },
      enc.min_key_frame_interval as usize,
      enc.max_key_frame_interval as usize,
    );

    Self {
      detector,
      bit_depth: u8::try_from(enc.bit_depth)
        .ok()
        .and_then(NonZeroU8::new)
        .expect("bit depth is validated by the encoder config"),
      frames: Vec::new(),
    }
  }

  fn analyze_next_frame(
    &mut self, frame_set: &[&Arc<Frame<T>>], input_frameno: usize,
    previous_keyframe: usize,
  ) -> (bool, Option<ScenecutResult>) {
    // The lookahead window only moves forward, so frames that have left it
    // will not be analyzed again.
    self.frames.retain(|cached| {
      frame_set.iter().any(|&frame| Arc::ptr_eq(frame, &cached.source))
    });

    let mut converted = Vec::with_capacity(frame_set.len());
    for &frame in frame_set {
      let cached =
        self.frames.iter().find(|cached| Arc::ptr_eq(&cached.source, frame));
      let luma = match cached {
        Some(cached) => Arc::clone(&cached.luma),
        None => {
          let luma = Arc::new(copy_luma(frame, self.bit_depth));
          self.frames.push(CachedFrame {
            source: Arc::clone(frame),
            luma: Arc::clone(&luma),
          });
          luma
        }
      };
      converted.push(luma);
    }

    let converted: Vec<_> = converted.iter().collect();
    self.detector.analyze_next_frame(
      &converted,
      input_frameno,
      previous_keyframe,
    )
  }
}

/// Copies the luma plane of `frame`, including its padding, into a
/// monochrome `v_frame` 0.5 frame with the same geometry.
fn copy_luma<T: Pixel, P: ScPixel + CastFromPrimitive<T>>(
  frame: &Frame<T>, bit_depth: NonZeroU8,
) -> ScFrame<P> {
  let src = &frame.planes[0];
  let cfg = &src.cfg;

  let mut luma = FrameBuilder::new(
    NonZeroUsize::new(cfg.width).expect("frame width is nonzero"),
    NonZeroUsize::new(cfg.height).expect("frame height is nonzero"),
    ChromaSubsampling::Monochrome,
    bit_depth,
  )
  .luma_padding_left(cfg.xorigin)
  .luma_padding_right(cfg.stride - cfg.xorigin - cfg.width)
  .luma_padding_top(cfg.yorigin)
  .luma_padding_bottom(cfg.alloc_height - cfg.yorigin - cfg.height)
  .build::<P>()
  .expect("pixel type matches the bit depth");

  // Both planes have the same stride and padded height, so their buffers
  // line up element for element.
  let dst = luma.y_plane.data_mut();
  assert_eq!(dst.len(), src.data.len());
  for (dst, &src) in dst.iter_mut().zip(src.data.iter()) {
    *dst = P::cast_from(src);
  }

  luma
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::frame::FrameAlloc;

  fn padded_frame<T: Pixel>(
    w: usize, h: usize, luma: impl Fn(usize, usize) -> u16,
  ) -> Frame<T> {
    let mut frame = Frame::<T>::new(w, h, ChromaSampling::Cs420);
    let plane = &mut frame.planes[0];
    let (stride, xorigin, yorigin) =
      (plane.cfg.stride, plane.cfg.xorigin, plane.cfg.yorigin);
    for y in 0..h {
      for x in 0..w {
        plane.data[(yorigin + y) * stride + xorigin + x] =
          T::cast_from(luma(x, y));
      }
    }
    plane.pad(w, h);
    frame
  }

  fn check_copy<T: Pixel, P: ScPixel + CastFromPrimitive<T> + Into<u32>>(
    bit_depth: u8,
  ) {
    // rav1e allocates planes with their dimensions rounded up to a multiple
    // of 8 and fills the extra area from `pad`. The copy keeps that geometry,
    // matching what scene detection saw when it read rav1e frames directly.
    let (w, h) = (37, 21);
    let frame =
      padded_frame::<T>(w, h, |x, y| ((x * 7 + y * 13) % 256) as u16);
    let luma: ScFrame<P> =
      copy_luma(&frame, NonZeroU8::new(bit_depth).unwrap());

    let src = &frame.planes[0];
    let geometry = luma.y_plane.geometry();
    assert!(luma.u_plane.is_none() && luma.v_plane.is_none());
    assert_eq!(luma.bit_depth.get(), bit_depth);
    assert_eq!((geometry.width.get(), geometry.height.get()), (40, 24));
    assert_eq!(
      (geometry.width.get(), geometry.height.get()),
      (src.cfg.width, src.cfg.height)
    );
    assert_eq!(geometry.stride.get(), src.cfg.stride);
    assert_eq!(
      luma.y_plane.data_origin(),
      src.cfg.yorigin * src.cfg.stride + src.cfg.xorigin
    );

    // Both the visible pixels and the edge-extended padding are preserved.
    assert_eq!(luma.y_plane.data().len(), src.data.len());
    for (&a, &b) in luma.y_plane.data().iter().zip(src.data.iter()) {
      assert_eq!(a.into(), u32::cast_from(b));
    }
  }

  #[test]
  fn copy_luma_u8() {
    check_copy::<u8, u8>(8);
  }

  #[test]
  fn copy_luma_u16() {
    check_copy::<u16, u16>(10);
  }

  #[test]
  fn copy_luma_8bit_in_u16() {
    check_copy::<u16, u8>(8);
  }

  /// Runs detection over a clip with a hard cut at frame `CUT`, feeding the
  /// detector the same sliding window that `ContextInner` does.
  fn detect<T: Pixel>(
    bit_depth: usize,
  ) -> Vec<(bool, Option<ScenecutResult>)> {
    const CUT: usize = 15;
    let (w, h) = (128, 96);
    let mut enc = EncoderConfig::with_speed_preset(6);
    enc.width = w;
    enc.height = h;
    enc.bit_depth = bit_depth;
    enc.speed_settings.scene_detection_mode = SceneDetectionSpeed::Standard;

    let frames: Vec<Arc<Frame<T>>> = (0..30)
      .map(|t| {
        Arc::new(padded_frame(w, h, |x, y| {
          let level = if t < CUT {
            // A smooth texture panning right.
            ((x + 2 * t) * 3 + y * 5) % 200 + 20
          } else {
            // A different scene: a checkerboard panning down.
            ((y + t) / 8 + x / 8) % 2 * 180 + 40
          };
          (level << (bit_depth - 8)) as u16
        }))
      })
      .collect();

    let window = InterConfig::new(&enc).keyframe_lookahead_distance() as usize;
    let mut detector = KeyframeDetector::<T>::new(&enc);
    let mut previous_keyframe = 0;
    let mut results = Vec::new();
    for input_frameno in 1..frames.len() {
      let end = (input_frameno + window).min(frames.len());
      let frame_set: Vec<_> = frames[input_frameno - 1..end].iter().collect();
      let result = detector.analyze_next_frame(
        &frame_set,
        input_frameno,
        previous_keyframe,
      );
      if result.0 {
        previous_keyframe = input_frameno;
      }
      results.push(result);

      // The cache holds exactly the frames in the current window.
      let cached = match &detector.inner {
        Inner::U8(d) => d.frames.len(),
        Inner::U16(d) => d.frames.len(),
      };
      assert_eq!(cached, frame_set.len());
    }

    assert!(results[CUT - 1].0, "missed the cut at frame {CUT}");
    results
  }

  #[test]
  fn detect_8bit_in_u16_matches_u8() {
    let u8_results = detect::<u8>(8);
    let u16_results = detect::<u16>(8);
    // `ScenecutResult` has no `PartialEq`, so compare the debug output.
    assert_eq!(format!("{u8_results:?}"), format!("{u16_results:?}"));
  }

  #[test]
  fn detect_high_bit_depth() {
    detect::<u16>(10);
  }
}
