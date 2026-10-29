//! Data structures for describing inline data in compiled machine code.

use crate::obj::ELF_WASMTIME_DATAMAP;
use crate::prelude::*;
use object::SectionKind;
use object::write::{Object, StandardSegment};
use std::ops::Range;

/// Builder for the inline-data map section of a Wasmtime compilation image.
///
/// The section starts with a zero `u32` format version and a `u32` entry count.
/// Each entry is a pair of little-endian `u32` values describing a half-open
/// byte range relative to the start of the text section. Entries are sorted and
/// overlapping or adjacent ranges are coalesced.
#[derive(Default)]
pub struct DataMapSection {
    ranges: Vec<Range<u32>>,
}

impl DataMapSection {
    /// Adds inline-data ranges for a function in the text section.
    ///
    /// The ranges are relative to `func.start`, must be sorted, and must be
    /// contained within `func`.
    pub fn push(&mut self, func: Range<u64>, ranges: &[Range<u32>]) {
        let func_start = u32::try_from(func.start).unwrap();
        let func_end = u32::try_from(func.end).unwrap();
        let func_len = func_end.checked_sub(func_start).unwrap();
        for range in ranges {
            assert!(range.start <= range.end && range.end <= func_len);
            if range.is_empty() {
                continue;
            }
            self.ranges
                .push(func_start + range.start..func_start + range.end);
        }
    }

    /// Adds inline-data ranges whose offsets are already relative to the start
    /// of the text section.
    pub fn push_text_ranges(&mut self, ranges: &[Range<u32>]) {
        self.ranges
            .extend(ranges.iter().filter(|range| !range.is_empty()).cloned());
    }

    /// Encodes this section into the object provided.
    pub fn append_to(self, obj: &mut Object) {
        let section = obj.add_section(
            obj.segment_name(StandardSegment::Data).to_vec(),
            ELF_WASMTIME_DATAMAP.as_bytes().to_vec(),
            SectionKind::ReadOnlyData,
        );
        obj.append_section_data(section, &self.finish(), 1);
    }

    fn finish(mut self) -> Vec<u8> {
        self.ranges.sort_by_key(|range| range.start);
        let mut merged: Vec<Range<u32>> = Vec::with_capacity(self.ranges.len());
        for range in self.ranges {
            assert!(range.start <= range.end);
            if range.is_empty() {
                continue;
            }
            if let Some(last) = merged.last_mut()
                && range.start <= last.end
            {
                last.end = last.end.max(range.end);
            } else {
                merged.push(range);
            }
        }

        let entries = u32::try_from(merged.len()).unwrap();
        let mut ret = Vec::with_capacity(8 + merged.len() * 8);
        ret.extend_from_slice(&0u32.to_le_bytes());
        ret.extend_from_slice(&entries.to_le_bytes());
        for range in merged {
            ret.extend_from_slice(&range.start.to_le_bytes());
            ret.extend_from_slice(&range.end.to_le_bytes());
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_and_encodes_ranges() {
        let mut section = DataMapSection::default();
        section.push(0x100..0x200, &[0x10..0x20, 0x20..0x30, 0x40..0x40]);
        section.push_text_ranges(&[0x118..0x140, 0x50..0x60, 0x70..0x70]);
        let bytes = section.finish();
        let expected = [
            0, 0, 0, 0, // version
            2, 0, 0, 0, // entry count
            0x50, 0, 0, 0, 0x60, 0, 0, 0, // 0x50..0x60
            0x10, 0x1, 0, 0, 0x40, 0x1, 0, 0, // 0x110..0x140
        ];
        assert_eq!(bytes, expected);
    }
}
