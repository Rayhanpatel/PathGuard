import streamlit as st

st.set_page_config(
    page_title="PathGuard — Spatial Safety Intelligence",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ PathGuard")
st.subheader("On-Device Spatial Safety Intelligence for Construction Workers")

st.markdown("""
PathGuard combines **always-on spatial hazard detection** with **on-device vision-language scene understanding** 
to protect construction workers — deployable on a Raspberry Pi 5, iPhone, or Android phone.

---

### Two Integrated Systems

| System | What It Does | How to Launch |
|--------|-------------|---------------|
| **🛡️ PathGuard HUD** | Real-time corridor-based obstacle detection with GroundedDINO + Depth Anything V2 | Sidebar → *PathGuard HUD* |
| **🌵 Cactus Narrator** | On-device VLM video narration with hybrid cloud routing + dynamic DINO prompt generation | Sidebar → *Cactus Narrator* |

### How They Connect

```
Cactus Narrator (scene understanding)
    │
    ├──► Dynamic DINO Prompts ──► PathGuard (zero-shot detection)
    │
    └──► RCP Telemetry ◄── PathGuard spatial data (corridor, depth, trip risk)
```

**Select a page from the sidebar** to get started.

---
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🛡️ PathGuard HUD")
    st.markdown("""
    - Trapezoidal corridor ROI for walking path
    - Classical CV fallback (every frame, no GPU needed)
    - GroundedDINO zero-shot detection (scheduled)
    - Depth Anything V2 for NEAR/MID/FAR urgency
    - State machine: CLEAR → OBSTACLE AHEAD → TRIP RISK
    - Audio alarm on trip hazards
    """)

with col2:
    st.markdown("#### 🌵 Cactus Narrator")
    st.markdown("""
    - On-device VLM (LFM2.5-VL-1.6B on Apple Neural Engine)
    - Hybrid routing: local NPU → Gemini cloud fallback
    - Dynamic GroundedDINO prompt generation from scene
    - RCP construction site telemetry extraction
    - Works on Raspberry Pi 5, iPhone 17 Pro, Pixel 6a
    """)

st.divider()
st.caption("UMD x Ironsite Startup Shell Hackathon 2026 | Spatial Intelligence")
