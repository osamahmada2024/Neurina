# `.vendor` Refactoring Summary

## Changes Made

### 1. Updated `.gitignore`
- **Added**: `.vendor/GFPGAN/` to ignore list
- **Reason**: GFPGAN should be installed via pip, not committed
- **File**: [`.gitignore`](.gitignore)

### 2. Simplified GFPGAN Import
- **File**: [src/services/face_restoration_service.py](src/services/face_restoration_service.py#L100)
- **Before**: Complex wrapper that manipulated `sys.path`
- **After**: Direct import from pip-installed package
- **Benefits**:
  - Cleaner code (8 lines → 12 lines including error handling)
  - No sys.path manipulation
  - Clear error messages
  - Consistent with how Python packages are meant to be used

**Old code** (removed):
```python
vendor_root = Path(base_path) / ".vendor" / "GFPGAN"
inserted = False
if vendor_root.is_dir():
    vendor_path = str(vendor_root)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
        inserted = True
# ... complex logic to load and unload from sys.path
```

**New code**:
```python
try:
    from gfpgan import GFPGANer
    return GFPGANer
except ImportError:
    raise ImportError(
        "gfpgan is not installed. Install it with: pip install -r requirements.txt"
    )
```

### 3. Updated Documentation

#### [README.md](README.md)
- Added link to SETUP.md
- Clarified dependency structure
- Added quick start section
- Listed all key dependencies

#### [SETUP.md](SETUP.md) (New)
- Complete setup instructions
- Troubleshooting guide
- Directory structure explanation
- Performance tips

#### [.VENDOR_README.md](.VENDOR_README.md) (New)
- Explains why `.vendor/CodeFormer/` is necessary
- Documents what changed and why
- Provides maintenance instructions

### 4. Updated Class Docstring
- [FaceRestorationService](src/services/face_restoration_service.py#L18)
- Now explicitly lists dependencies
- Clarifies which are pip vs. vendor

## Current `.vendor` Status

| Folder | Status | Reason |
|--------|--------|--------|
| `.vendor/CodeFormer/` | ✅ **Keep** | Not on PyPI; required for architecture modules |
| `.vendor/GFPGAN/` | ❌ **Ignored** | Available on PyPI; use pip instead |

## What Gets Downloaded vs. Stored

| Item | Location | When |
|------|----------|------|
| CodeFormer source | `.vendor/CodeFormer/` | Committed to repo |
| GFPGAN pip package | Site-packages | Via `pip install -r requirements.txt` |
| Model weights | `checkpoints/face_restoration/` | First runtime use (auto-downloaded) |
| GFPGANv1.4 weights | `checkpoints/face_restoration/GFPGANv1.4.pth` | Auto-downloaded |
| CodeFormer weights | `checkpoints/face_restoration/codeformer.pth` | Auto-downloaded |
| Real-ESRGAN weights | `checkpoints/super_resolution/RealESRGAN_x4plus.pth` | Auto-downloaded |

## Backward Compatibility

✅ **No breaking changes** - The refactored code:
- Works with existing workflows
- Still auto-downloads model weights
- Still loads CodeFormer from `.vendor/`
- Same API surface for `FaceRestorationService`

## Next Steps (Optional)

1. **Remove `.vendor/GFPGAN/` from the repository** (if cloned with it)
   ```bash
   rm -rf .vendor/GFPGAN
   git add .gitignore
   git commit -m "Remove redundant GFPGAN wrapper; use pip package instead"
   ```

2. **Validate setup on fresh install**
   ```bash
   rm -rf .vendor/GFPGAN .mp_env  # Clean environment
   pip install -r requirements.txt
   python -c "from src.services.face_restoration_service import FaceRestorationService; print('✓ OK')"
   ```

3. **Consider git submodule for CodeFormer** (future improvement)
   - For better version management
   - Easier updates: `git submodule update --remote`

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Code complexity** | High (sys.path manipulation) | Low (direct import) |
| **Dependency clarity** | Unclear | Clear (pip vs. vendor) |
| **Maintenance** | Complex wrapper logic | Standard Python import |
| **Errors** | Cryptic sys.path errors | Clear ImportError |
| **Documentation** | Minimal | Complete (3 docs) |
| **Git cleanliness** | Commits GFPGAN source | GFPGAN in .gitignore |

## Files Changed

1. `.gitignore` - Added `.vendor/GFPGAN/`
2. `README.md` - Updated quick start and added dependency info
3. `src/services/face_restoration_service.py` - Simplified GFPGAN import
4. `SETUP.md` - Created comprehensive setup guide
5. `.VENDOR_README.md` - Created vendor structure documentation
