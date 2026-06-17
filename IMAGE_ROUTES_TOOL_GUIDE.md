# ImageRoutesTool - Multi-Path Image Upload

## 🎯 **الـ Tool يدعم كل الحالات:**

### **Reference Image Paths:**

```python
# Path 1: من الـ Web Search (URL)
reference_id = tool.upload_reference_from_url(
    "https://example.com/celebrity.jpg"
)
# ✓ ReferenceSelectorAgent يستخدمها

# Path 2: من Bytes (AI-generated أو معالج)
reference_id = tool.upload_reference_from_bytes(
    image_bytes=processed_image,
    filename="enhanced_reference.jpg"
)
# ✓ لو كان عندك preprocessing للـ reference
```

---

### **Source Image Paths:**

```python
# Path 1: من URL (لو اليوزر أعطى URL بدل file)
source_id = tool.upload_source_from_url(
    "https://user-storage.com/my-photo.jpg"
)
# ✓ Alternative للـ manual upload

# Path 2: من Bytes (صورة معالجة أو مصدر آخر)
source_id = tool.upload_source_from_bytes(
    image_bytes=user_image_bytes,
    filename="user_photo.jpg"
)
# ✓ لو كانت الصورة موجودة في memory

# Path 3: User Upload (الـ default - عبر الـ frontend)
# POST /api/v1/images/upload (مباشرة)
# ✓ ما يحتاج tool - عبر الـ routes الموجودة
```

---

### **Generic Methods (أي type):**

```python
# استخدم الـ type parameter
image_id = tool.upload_image_from_url(
    image_url="...",
    image_type="source"  # أو "reference"
)

image_id = tool.upload_image_from_bytes(
    image_bytes=data,
    image_type="reference",  # أو "source"
    filename="image.jpg"
)
```

---

## 📊 **الـ Complete Scenarios:**

### **Scenario 1: Web Reference + User Source (Standard)**
```
User: "I want Tamer Hosny style"
  ↓
Agent Search → Select → Upload from Web URL
  ↓
reference_id ← tool.upload_reference_from_url()
  ↓
User uploads source file from device
  ↓
source_id ← POST /api/v1/images/upload
  ↓
Translate(source_id, reference_id)
```

### **Scenario 2: Web Reference + URL Source**
```
Agent gets both from web/API
  ↓
reference_id ← tool.upload_reference_from_url()
source_id ← tool.upload_source_from_url()
  ↓
Translate(source_id, reference_id)
```

### **Scenario 3: AI-Generated Images**
```
Agent generates both (Bytes)
  ↓
reference_id ← tool.upload_reference_from_bytes()
source_id ← tool.upload_source_from_bytes()
  ↓
Translate(source_id, reference_id)
```

### **Scenario 4: Mixed Sources**
```
Web reference + AI source
  ↓
reference_id ← tool.upload_reference_from_url()
source_id ← tool.upload_source_from_bytes()
  ↓
Translate(source_id, reference_id)
```

---

## 🔧 **How to Use in Agents:**

### **ReferenceSelectorAgent (يختار من الـ web):**
```python
selected_url = await self._select_best_image(...)
reference_id = tool.upload_reference_from_url(selected_url)
```

### **Future: SourceProcessingAgent (لو كانت موجودة):**
```python
if image_source == "url":
    source_id = tool.upload_source_from_url(source_url)
elif image_source == "bytes":
    source_id = tool.upload_source_from_bytes(source_bytes)
else:  # file from user
    # user uploads via frontend
    pass
```

---

## ✅ **Summary:**

**الـ ImageRoutesTool الآن تدعم:**
- ✅ Reference من URL (web search)
- ✅ Reference من Bytes (preprocessing, AI)
- ✅ Source من URL (external)
- ✅ Source من Bytes (memory, AI)
- ✅ Generic uploads (أي type)
- ✅ Translation triggering

**المرونة الكاملة لـ أي مسار!** 🚀
