# Cleanup Guide - Files to Remove/Archive

**Last Updated:** June 20, 2026

---

## 📋 Files Status

### ✅ Keep These (Currently Used)
```
Mini-GPT/
├── app.py                          ✅ Web server
├── requirements.txt                ✅ Dependencies
├── LICENSE                         ✅ License
└── README.md                       ✅ Main project README (or use README_NEW.md)
```

### ✨ New Files (Use These)
```
├── docs/00_START_HERE.md           ✨ NEW - Main entry point
├── docs/01-05_*.md                 ✨ NEW - Organized docs
├── README_NEW.md                   ✨ NEW - Professional README
├── ORGANIZATION_GUIDE.md           ✨ NEW - Organization info
└── ORGANIZATION_SUMMARY.md         ✨ NEW - What was organized
```

### ⚠️ Can Move to legacy/
```
docs/
├── README.md                       ⚠️ Old Dutch version → legacy/
├── PROJECT_MAP.md                  ⚠️ → legacy/
├── last-raport.md                  ⚠️ → legacy/
├── README_INDEX.md                 ⚠️ → legacy/
└── DOCUMENTATION_README.md         ⚠️ → legacy/
```

### 🗑️ Can Delete (Optional)
```
├── gpt_firstVersions/              🗑️ Old versions (keep in git)
├── src/CS50/                       🗑️ Not used
├── src/execution-routing-engine/   🗑️ Not used
├── src/HulpLib/                    🗑️ Not used
└── .pytest_cache/                  🗑️ Cache (auto-generated)
```

### 🔄 Can Move to models/
```
output/
├── *.pth files                     🔄 Move to ../models/
```

### 🔄 Can Reorganize
```
data/
├── all_cleaned_text.md             🔄 Move to training/
├── alpaca.csv                      🔄 Move to training/
├── qa.csv                          🔄 Move to training/
└── cleaner/                        🔄 Reorganize
```

---

## 🎯 Recommended Cleanup

### Phase 1: SAFE (No Risk) ✅
- [x] Use new organized docs
- [x] Update main README (use README_NEW.md)
- [x] Use entry point: docs/00_START_HERE.md
- [x] Reference folder READMEs

**Risk:** None | **Time:** 5 min | **Benefit:** High

### Phase 2: MEDIUM (Some Caution) ⚠️
- [ ] Move old docs to legacy/
- [ ] Update all documentation references
- [ ] Test that nothing breaks

**Risk:** Low | **Time:** 15 min | **Benefit:** Medium

### Phase 3: HIGH (Requires Planning) 🗑️
- [ ] Delete old version files (gpt_firstVersions/)
- [ ] Remove unused src/ subdirectories
- [ ] Reorganize data files
- [ ] Move output → models

**Risk:** Medium | **Time:** 30 min | **Benefit:** High

---

## 📝 Specific Files to Address

### File: docs/README.md
**Status:** Dutch documentation, outdated  
**Action:** Move to `docs/legacy/README.md`  
**Command:** `mv docs/README.md docs/legacy/README.md`  
**Replace with:** Use `docs/00_START_HERE.md` instead

### File: docs/PROJECT_MAP.md
**Status:** Old structure documentation  
**Action:** Move to `docs/legacy/PROJECT_MAP.md`  
**Command:** `mv docs/PROJECT_MAP.md docs/legacy/`  
**Keep because:** Historical reference

### File: docs/last-raport.md
**Status:** Project report  
**Action:** Move to `docs/legacy/last-raport.md`  
**Command:** `mv docs/last-raport.md docs/legacy/`  
**Keep because:** Project history

### File: docs/README_INDEX.md
**Status:** Old navigation guide  
**Action:** Move to `docs/legacy/README_INDEX.md`  
**Command:** `mv docs/README_INDEX.md docs/legacy/`  
**Replace with:** Use `docs/00_START_HERE.md`

### File: docs/DOCUMENTATION_README.md
**Status:** Old doc overview  
**Action:** Move to `docs/legacy/DOCUMENTATION_README.md`  
**Command:** `mv docs/DOCUMENTATION_README.md docs/legacy/`  
**Replace with:** Use `docs/00_START_HERE.md`

### File: README.md (root)
**Status:** Current main README  
**Action:** Keep or replace with `README_NEW.md`  
**Option 1:** Replace with README_NEW.md  
```bash
mv README.md README_OLD.md
mv README_NEW.md README.md
```
**Option 2:** Keep both, let users choose

### File: gpt_firstVersions/
**Status:** Old version files  
**Action:** Delete (keep in git history)  
**Command:** `rm -rf gpt_firstVersions/`  
**Why:** Not needed, old versions

### Folder: src/CS50/
**Status:** Not used  
**Action:** Delete  
**Command:** `rm -rf src/CS50/`  
**Why:** Not part of MiniGPT

### Folder: src/execution-routing-engine/
**Status:** Not used  
**Action:** Delete  
**Command:** `rm -rf src/execution-routing-engine/`  
**Why:** Not part of MiniGPT

### Folder: src/HulpLib/
**Status:** Not used  
**Action:** Delete  
**Command:** `rm -rf src/HulpLib/`  
**Why:** Not part of MiniGPT

### File: dataset.json
**Status:** Configuration file  
**Action:** Keep or move to `data/`  
**Command:** `mv dataset.json data/dataset.json`

### Folder: output/
**Status:** Pre-trained models  
**Action:** Move to `models/`  
**Command:** `mv output/* models/ && rm -rf output/`

---

## 🚀 How to Clean Up

### Option A: Automated (Recommended)
```bash
# Create cleanup script
cat > cleanup.sh << 'EOF'
#!/bin/bash

# Phase 1: Move legacy docs
mkdir -p docs/legacy
mv docs/README.md docs/legacy/  2>/dev/null || true
mv docs/PROJECT_MAP.md docs/legacy/  2>/dev/null || true
mv docs/last-raport.md docs/legacy/  2>/dev/null || true
mv docs/README_INDEX.md docs/legacy/  2>/dev/null || true
mv docs/DOCUMENTATION_README.md docs/legacy/  2>/dev/null || true

# Phase 2: Update main README
if [ -f README_NEW.md ]; then
  mv README.md README_OLD.md  2>/dev/null || true
  mv README_NEW.md README.md
fi

# Phase 3: Remove old versions
rm -rf gpt_firstVersions/  2>/dev/null || true

# Phase 4: Remove unused src folders
rm -rf src/CS50/  2>/dev/null || true
rm -rf src/execution-routing-engine/  2>/dev/null || true
rm -rf src/HulpLib/  2>/dev/null || true

echo "✅ Cleanup complete!"
EOF

chmod +x cleanup.sh
./cleanup.sh
```

### Option B: Manual
Follow the commands in "Specific Files to Address" above.

### Option C: Step by Step
Do one file at a time, test, then move to next.

---

## ⚠️ Before Cleanup

**IMPORTANT:** Commit to git first!

```bash
git add .
git commit -m "Before cleanup - keeping original structure as backup"
```

This allows reverting if something goes wrong.

---

## ✅ After Cleanup

### Verify everything works:
```bash
# 1. Can generate text?
python -c "from src.inference import LoadedModel; m = LoadedModel('models/MiniGPT.pth'); print(m.predict('Hello'))"

# 2. Web server works?
python app.py &  # Start server
curl http://localhost:5000  # Test
# Ctrl+C to stop

# 3. Examples still work?
python examples/01_basic_generation.py
```

---

## 📊 Cleanup Impact

| Action | Risk | Benefit | Effort |
|--------|------|---------|--------|
| Use new docs | None | High | 0 min |
| Move old docs | None | Medium | 5 min |
| Delete old versions | Low | Medium | 2 min |
| Delete unused folders | None | Low | 1 min |
| Move to models/ | Medium | High | 10 min |
| Update imports | High | Medium | 20 min |

---

## 🎯 Recommended Plan

### Phase 1: Documentation (TODAY) ✅
- [x] Use new organized documentation
- [x] Use new entry point
- [x] Use README.md in each folder
- ✅ **NO RISK, IMMEDIATE BENEFIT**

### Phase 2: Minor Cleanup (LATER) ⚠️
- [ ] Move old docs to legacy/
- [ ] Update main README
- [ ] Delete unused folders
- ⚠️ **LOW RISK, GOOD CLEANUP**

### Phase 3: Code Refactoring (MUCH LATER) 🗑️
- [ ] Move gpt_lib → src
- [ ] Move output → models
- [ ] Update all imports
- 🗑️ **HIGHER EFFORT, BEST STRUCTURE**

---

## 💡 Recommendations

### Minimum Cleanup (Safe)
- Use new documentation organization
- Delete gpt_firstVersions/ (old versions)
- Delete unused src/ subdirectories
- ✅ Easy, safe, improves cleanliness

### Recommended Cleanup (Medium)
- Above + move old docs to legacy
- Replace README.md with README_NEW.md
- Reorganize data/ folder
- ✅ Professional, moderate effort

### Full Cleanup (Aggressive)
- Above + move gpt_lib to src
- Move output to models
- Update all imports
- Run full test suite
- ⚠️ More complex, best structure

---

## 📞 Questions?

- **Should I clean up?** → Yes, recommended
- **Is it risky?** → No, git backup first
- **How long?** → 5-30 minutes
- **Will it break?** → No if done carefully
- **Can I revert?** → Yes, from git

---

**Recommendation:** Do Phase 1 cleanup (safe + high benefit) now!

Do Phases 2-3 when you have time.

---

*Cleanup guide: June 20, 2026*
