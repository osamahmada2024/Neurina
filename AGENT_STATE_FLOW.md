# Agent State Flow - Data Schema

## 🔄 **الـ Complete Data Flow:**

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: User Input                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input:                                                         │
│  {                                                              │
│    "user_input": "I want Tamer Hosny style",                   │
│    "user_id": "507f1f77bcf86cd799439011",                      │
│    "auth_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",                │
│    "errors": []                                                 │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: QueryAgent                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: QueryAgentState                                         │
│  - user_input: str                                              │
│  - user_id: str                                                 │
│  - auth_token: str                                              │
│                                                                 │
│  Process: Extract search query using LLM                        │
│                                                                 │
│  Output: QueryAgentState + {                                    │
│    "generated_search_query": "Tamer Hosny male celebrity..."  │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: APIExecutionAgent                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: APIExecutionAgentState                                  │
│  - generated_search_query: str                                  │
│  - (all previous fields)                                        │
│                                                                 │
│  Process: Search web using Serper API                           │
│                                                                 │
│  Output: APIExecutionAgentState + {                             │
│    "candidate_images": {                                        │
│      "img_id_1": "https://example.com/photo1.jpg",             │
│      "img_id_2": "https://example.com/photo2.jpg",             │
│      "img_id_3": "https://example.com/photo3.jpg"              │
│    }                                                            │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: QualityControlAgent                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: QualityControlAgentState                                │
│  - candidate_images: Dict[str, str]                             │
│  - (all previous fields)                                        │
│                                                                 │
│  Process: Score quality (face count, blur, resolution)          │
│           Filter by threshold                                   │
│                                                                 │
│  Output: QualityControlAgentState + {                           │
│    "quality_score": "0.87",                                     │
│    "candidate_images": {  // filtered                           │
│      "img_id_1": "https://example.com/photo1.jpg",             │
│      "img_id_2": "https://example.com/photo2.jpg"              │
│    }                                                            │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: ReferenceSelectorAgent                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: ReferenceSelectorAgentState                             │
│  - candidate_images: Dict[str, str]  // filtered                │
│  - (all previous fields)                                        │
│                                                                 │
│  Process: Use LLM to pick best image                            │
│           Upload to backend via ImageRoutesTool                 │
│                                                                 │
│  Output: ReferenceSelectorAgentState + {                        │
│    "selected_reference_url": "https://example.com/photo1.jpg", │
│    "reference_image_id": "64a3b8c9d0e1f2g3h4i5j6k7l8"         │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Workflow Complete (Agents)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Return SupervisorAgentState:                                   │
│  {                                                              │
│    "user_input": "I want Tamer Hosny style",                   │
│    "user_id": "507f1f77bcf86cd799439011",                      │
│    "generated_search_query": "Tamer Hosny male...",            │
│    "quality_score": "0.87",                                     │
│    "selected_reference_url": "https://...",                    │
│    "reference_image_id": "64a3b8c9...",                        │
│    "errors": []                                                 │
│  }                                                              │
│                                                                 │
│  ✓ Send to frontend:                                            │
│    "Reference ready! Upload your source image"                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: User Uploads Source (Manual - Frontend)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  POST /api/v1/images/upload                                    │
│  {file: source_photo.jpg, type: "source"}                      │
│                                                                 │
│  Response: {source_image_id: "64a3b8c9..."}                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 8: Translation                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  POST /api/v1/images/translate                                 │
│  {                                                              │
│    "source_image_id": "64a3b8c9...",                           │
│    "reference_image_id": "64a3b8c9..."                         │
│  }                                                              │
│                                                                 │
│  Response: {translated_image_id: "64a3b8c9..."}               │
│                                                                 │
│  ✓ Final Image: https://cloudinary.com/final_style.jpg       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 **State Schemas Summary:**

| Agent | Input State | Output Fields | Count |
|-------|------------|---------------|-------|
| **Query** | base (user_input) | + generated_search_query | 4 fields |
| **API** | QueryAgentState | + candidate_images | 5 fields |
| **QC** | APIExecutionAgentState | + quality_score | 6 fields |
| **Selector** | QualityControlAgentState | + selected_reference_url + reference_image_id | 8 fields |
| **Supervisor** | ReferenceSelectorAgentState | + all above (complete) | 12 fields |

---

## ✅ **Benefits:**

✓ **Type Safe** - Know exactly what each agent receives/produces  
✓ **Clear Contract** - Each agent has defined inputs/outputs  
✓ **Easy to Debug** - Track state at each step  
✓ **Documentation** - Code self-documents the flow  
✓ **IDE Support** - Auto-complete for state fields  
✓ **Maintainable** - Clear data structure evolution
