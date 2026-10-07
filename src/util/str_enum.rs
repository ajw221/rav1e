// Copyright (c) 2026, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

/// Implements `FromStr`, `Display` and `variants()` for a fieldless enum.
///
/// Parsing is ASCII case-insensitive and also accepts the aliases listed
/// after a variant (`Variant | "alias"`); `Display` prints the variant name.
macro_rules! impl_str_enum {
  ($name:ident { $($variant:ident $(| $alias:literal)*),+ $(,)? }) => {
    impl ::std::str::FromStr for $name {
      type Err = String;

      fn from_str(s: &str) -> Result<Self, Self::Err> {
        $(
          if s.eq_ignore_ascii_case(stringify!($variant))
            $(|| s.eq_ignore_ascii_case($alias))*
          {
            return Ok($name::$variant);
          }
        )+
        Err(format!("valid values: {}", $name::variants().join(", ")))
      }
    }

    impl ::std::fmt::Display for $name {
      fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        f.write_str(match self {
          $($name::$variant => stringify!($variant),)+
        })
      }
    }

    impl $name {
      /// Returns an array of valid values which can be converted into this enum.
      #[allow(dead_code)]
      pub fn variants(
      ) -> [&'static str; [$(stringify!($variant) $(, $alias)*),+].len()] {
        [$(stringify!($variant) $(, $alias)*),+]
      }
    }
  };
}

pub(crate) use impl_str_enum;
