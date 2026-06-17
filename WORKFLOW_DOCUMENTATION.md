# الـ Complete Style Transfer Flow

## 🔄 **الـ Correct Workflow:**

### **Step 1: اليوزر يطلب style (Automatic - Backend)**

```bash
POST /api/v1/style-transfer/request
Content-Type: application/json
Authorization: Bearer <token>

{
  "style_description": "I want my picture in Tamer Hosny style"
}
```

**Backend Response:**
```json
{
  "success": true,
  "reference_image_id": "507f1f77bcf86cd799439011",
  "quality_score": "0.87",
  "message": "Reference image selected! Please upload your source image."
}
```

**ماذا يحدث في الـ Backend:**
1. QueryAgent: استخراج الـ query من الـ user input
2. APIExecutionAgent: البحث عن صور من الإنترنت (Serper API)
3. QualityControlAgent: تصفية الصور بناء على الجودة
4. ReferenceSelectorAgent: 
   - اختيار أفضل صورة
   - **تحميل الصورة من الإنترنت**
   - **رفع الصورة للـ backend** ← `ImageRoutesTool.upload_reference_from_url()`
   - حفظ reference_image_id في DB
5. Return response مع reference_image_id

---

### **Step 2: اليوزر يرفع صورته (Manual - Frontend)**

في الـ Frontend، يتم عرض UI form:

```html
<form>
  <h3>Reference image selected! ✓</h3>
  <p>Now upload your source image:</p>
  
  <input type="file" accept="image/*" id="sourceImage" />
  <button onclick="uploadAndTranslate()">Transform Image</button>
</form>
```

**الـ JavaScript:**
```javascript
async function uploadAndTranslate() {
  const file = document.getElementById('sourceImage').files[0];
  const formData = new FormData();
  formData.append('file', file);
  formData.append('image_type', 'source');
  
  // Step 2.1: Upload source image
  const uploadRes = await fetch('/api/v1/images/upload', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
    body: formData
  });
  const { image_id: source_image_id } = await uploadRes.json();
  
  // Step 2.2: Trigger translation
  const translateRes = await fetch('/api/v1/images/translate', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
    body: JSON.stringify({
      source_image_id,
      reference_image_id, // من الخطوة الأولى
    })
  });
  const { translated_image_id } = await translateRes.json();
  
  // Step 2.3: Show result
  showTransformedImage(translated_image_id);
}
```

---

## 📊 **الـ Component Mapping:**

| Component | Role | Input | Output | Notes |
|-----------|------|-------|--------|-------|
| **ReferenceSelectorAgent** | يختار + يرفع reference من الإنترنت | URLs من الـ web search | reference_image_id | استخدم `ImageRoutesTool.upload_reference_from_url()` |
| **Frontend Form** | اليوزر يرفع source من جهازه | File من الجهاز | source_image_id | استخدم `POST /api/v1/images/upload` |
| **Translation Route** | يحول الأسلوب | Both IDs | translated_image_id | موجود في الـ routes |

---

## ✅ **Key Points:**

✓ **ReferenceSelectorAgent يرفع** (من الإنترنت)  
✓ **اليوزر يرفع** (من جهازه)  
✓ **معزولين تماماً** - كل واحد بـ upload path مختلف  
✓ **الـ Flow متسلسل** - Agent أولاً، ثم اليوزر
