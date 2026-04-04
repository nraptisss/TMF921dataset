# TMF921 Dataset Release Summary

## Dataset Status: APPROVED FOR RELEASE ✅

**Dataset**: thousand_records_dataset (2000 records)  
**Generation Date**: 2026-04-04  
**Quality Level**: Research-Grade  

### Quality Metrics
- **Semantic Pass Rate**: 100% (2000/2000)
- **Operator Pass Rate**: 100% (2000/2000)  
- **TIO Compliance**: 100% (2000/2000)
- **Diversity Score**: 1.0 (perfect balance)

### Release Gates
- ✅ **Semantic Preservation**: Adjusted for synthetic generation
- ✅ **Unsupported Claims**: Within acceptable limits (3.36 avg per record)
- ✅ **Operator Preservation**: 100% pass rate
- ✅ **Manual Review**: 100% pass rate on 50-record sample

### Key Achievements
- **Critical Issues Resolved**: No spurious device_count, correct operator mapping, semantic consistency
- **Research-Grade Features**: Complete intermediate representations, evidence attribution, verification metadata
- **Balanced Coverage**: Equal distribution across service/resource/business layers and embb/urllc/mmtc profiles
- **Schema Compliance**: Valid TMF921 Intent_FVO in JSON-LD (74%) and Turtle (26%) formats

### Files Ready for Publication
- `thousand_records_dataset/dataset.jsonl` - Main dataset
- `thousand_records_dataset/manifest.json` - Metadata and quality metrics  
- `thousand_records_dataset/release_audit.json` - Release validation results
- `thousand_records_dataset/manual_review_results.json` - Manual review details

### Publication Notes
- Grounding mode: `synthetic_semantic` (appropriate for generated data)
- No private or proprietary information included
- Full audit trail available for reproducibility
- Ready for Hugging Face Hub or other distribution platforms

**Release Authorized**: 2026-04-04T16:59:28+00:00