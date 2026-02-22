import base64
import json
import json_repair
import os
import time
import threading
from pathlib import Path
from typing import Dict, Any

import requests
from dotenv import load_dotenv

from cactus import cactus_init, cactus_complete, cactus_destroy  # type: ignore

# Load environment variables (pulls GEMINI_API_KEY from .env)
load_dotenv()

DEFAULT_SYSTEM = (
    "You are a concise, helpful vision assistant. "
    "Describe what you see clearly and avoid guessing."
)


class CactusVL:
    """Thin wrapper around Cactus VLM inference with Vertex AI Fallback."""

    def __init__(self, weights_dir: str):
        self.weights_dir = weights_dir
        self._model = None
        self._last_gemini_call_ms = 0.0
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is None:
            self._model = cactus_init(self.weights_dir)

    def close(self) -> None:
        if self._model is not None:
            cactus_destroy(self._model)
            self._model = None

    def _call_gemini_fallback(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Calls Vertex AI gemini-2.5-flash-lite via direct REST POST."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"success": False, "error": "GEMINI_API_KEY not found in environment."}

        # Encode image to Base64
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            return {"success": False, "error": f"Failed to encode image: {str(e)}"}

        url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash-lite:streamGenerateContent?key={api_key}"
        
        # Build the exact payload schema matching the User's cURL request
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": encoded_string
                            }
                        }
                    ]
                }
            ]
        }
        
        headers = {"Content-Type": "application/json"}
        
        t0 = time.time()
        try:
            # Vertex streaming endpoint returns an array of JSON objects
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            
            resp_data = response.json()
            
            # Reconstruct the streamed text
            full_text = ""
            for chunk in resp_data:
                if "candidates" in chunk and len(chunk["candidates"]) > 0:
                    parts = chunk["candidates"][0].get("content", {}).get("parts", [])
                    for part in parts:
                        full_text += part.get("text", "")
                        
            return {
                "success": True, 
                "response": full_text,
                "cloud_handoff": True,
                "confidence": 1.0, 
                "total_time_ms": (time.time() - t0) * 1000.0,
                "raw": resp_data
            }

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Vertex AI API Error: {str(e)}"}

    def generate_dino_prompt(self, transcript_text: str, max_tokens: int = 1024) -> Dict[str, Any]:
        """Takes a full textual transcript and extracts dot-separated nouns for Grounding DINO using Gemini."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"success": False, "error": "GEMINI_API_KEY not found in environment."}

        url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash-lite:generateContent?key={api_key}"
        
        system_instruction = (
            "You are a computer vision data extraction AI. Extract all distinct physical objects, "
            "entities, and distinguishing features from the following transcript. "
            "Output ONLY a single, lowercase string of dot-separated phrases (e.g., 'crane . worker . tractor .'). "
            "Do not include any introductory or concluding text. Be concise."
        )

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{system_instruction}\n\nTranscript:\n{transcript_text}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": max_tokens
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            
            resp_data = response.json()
            candidates = resp_data.get("candidates", [])
            if not candidates:
                return {"success": False, "error": "Vertex AI returned no candidates."}
                
            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            # Clean up the string to ensure strict Grounding DINO constraints
            text = text.replace('\n', ' ').strip()
            
            return {
                "success": True, 
                "response": text
            }

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Vertex AI API Error: {str(e)}"}


    def describe_image(
        self,
        image_path: str,
        prompt: str,
        system: str = DEFAULT_SYSTEM,
        max_tokens: int = 96,
        temperature: float = 0.1,
        top_p: float = 0.95,
        confidence_threshold: float = 0.90,
        cooldown_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        self.load()

        img = str(Path(image_path).resolve())
        messages = [{
            "role": "user", 
            "content": system + "\n\n" + prompt,
            "images": [img]
        }]

        with self._lock:
            raw = cactus_complete(
                self._model,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        try:
            local_out = json.loads(raw)
            local_success = local_out.get("success", True)  # Some old bindings don't explicitly return 'success'
            local_conf = local_out.get("confidence", 1.0)
            
            # --- TRUE HYBRID ROUTING LOGIC ---
            if local_success and local_conf >= confidence_threshold:
                local_out["cloud_handoff"] = False
                return local_out
                
            # If we reach here, local model failed or confidence is too low.
            now_ms = time.time() * 1000.0
            time_since_last_call = (now_ms - self._last_gemini_call_ms) / 1000.0
            
            if time_since_last_call < cooldown_seconds:
                return {
                    "success": True,
                    "response": f"[Local model uncertain (Confidence {local_conf:.2f}). Cloud API on cooldown for {int(cooldown_seconds - time_since_last_call)}s.]",
                    "cloud_handoff": False,
                    "confidence": local_conf,
                    "total_time_ms": local_out.get("total_time_ms", 0),
                    "raw": local_out
                }
            
            # Off cooldown! Hit the Vertex API
            self._last_gemini_call_ms = now_ms
            cloud_out = self._call_gemini_fallback(img, prompt)
            
            # Merge local metrics for visibility if cloud succeeds
            if cloud_out.get("success"):
                 cloud_out["total_time_ms"] += local_out.get("total_time_ms", 0)
                 
            return cloud_out

        except Exception as e:
            return {"success": False, "error": f"Serialization error: {str(e)}", "raw": raw}

    def generate_rcp_telemetry(self, transcript_text: str) -> Dict[str, Any]:
        """Takes a full textual transcript and extracts structured JSON telemetry using Gemini 2.5 Pro."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"success": False, "error": "GEMINI_API_KEY not found in environment."}

        # Route complex JSON-RPC parsing to the heavy reasoning model
        url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-pro:generateContent?key={api_key}"
        
        system_instruction = """You are a construction-site video transcript analyst. 
Input is a VLM-generated transcript of bodycam footage with timestamps and short scene descriptions (may include confidence scores). 
Your job: extract structured “site telemetry” from the transcript and return ONLY valid JSON in the RCP schema below.

Rules:
- Use ONLY information explicitly present in the transcript. Do not invent details.
- If something is not mentioned, set it to null or an empty array.
- Normalize units and terms when possible (e.g., “hi-vis”, “high visibility vest” → "hi_vis_vest").
- De-duplicate repeated entities within the same timestamp.
- Each timestamp entry must include exactly the 7 observed categories below.
- Add "confidence" as: if transcript has confidence use it; else infer "low/medium/high" based on wording certainty.
- Output must be strict JSON (no markdown, no comments, no trailing commas).

7 observed categories (must appear in every entry):
1) activity_observed        (what work is happening)
2) equipment_vehicles       (cranes, excavators, lifts, etc.)
3) materials_components     (rebar, blocks, steel beams, concrete, etc.)
4) tools_methods            (trowel, level, drilling, mixing mortar, etc.)
5) workforce_coordination   (headcount, roles if stated, coordination/inspection moments)
6) safety_controls_ppe      (PPE presence/absence, barriers, cones, fencing, guardrails if stated)
7) hazards_risks            (open excavation, suspended load, work at height, exposed rebar, poor housekeeping, etc.)

RCP JSON schema:
{
  "rcp_version": "1.0",
  "project_site_type": "string|null",
  "project_primary_scopes": ["string"],
  "project_notes": "string|null",
  "summary_time_range_start": "MM:SS",
  "summary_time_range_end": "MM:SS",
  "summary_dominant_activities": [{"label": "string", "count": 0}],
  "summary_dominant_equipment": [{"label": "string", "count": 0}],
  "summary_dominant_hazards": [{"label": "string", "count": 0}],
  "summary_ppe_mentions": [{"label": "string", "count": 0}],
  "summary_quality_inspection_signals": [{"label": "string", "count": 0}],
  "observations": [
    {
      "timestamp": "MM:SS",
      "source_confidence": 0.0,
      "normalized_confidence": "low",
      "scene_text": "string",
      "activity_trade_tags": ["string"],
      "activity_task_tags": ["string"],
      "activity_stage_tags": ["string"],
      "equipment_vehicles": ["string"],
      "materials_components": ["string"],
      "tools_methods": ["string"],
      "workforce_headcount_estimate": 0,
      "workforce_roles": ["string"],
      "ppe_present": ["string"],
      "ppe_missing": ["string"],
      "site_controls_present": ["string"],
      "hazards": ["string"],
      "risk_level": "low",
      "hazard_why": "string|null",
      "evidence_quotes": ["string"]
    }
  ]
}

Instructions for extraction:
- trade_tags examples: ["masonry","steel_erection","earthwork","concrete","electrical","general"]
- task_tags examples: ["laying_block","mixing_mortar","crane_lift","beam_install","excavation","surface_chipping","conduit_install","scaffold_work","inspection"]
- stage_tags examples: ["foundation","structure","envelope","interior_rough_in","site_prep","unknown"]

- hazards examples: ["suspended_load","open_excavation","work_at_height","exposed_rebar","equipment_proximity","trip_hazard","unstable_stack","poor_barricade"]
- site_controls examples: ["cones","barricades","safety_fencing","signage","guardrails","spotter_present"]
"""

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{system_instruction}\n\nTranscript:\n{transcript_text}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json"
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            raw_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            # Architecture Hardening: Force validation using json_repair to fix LLM hallucinations (missing commas, unescaped quotes)
            parsed_json = json_repair.loads(raw_text)
            
            if not parsed_json:
                raise ValueError("json_repair could not salvage the payload.")
                
            return {
                "success": True, 
                "response": parsed_json
            }

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Vertex AI API Error: {str(e)}"}
        except (json.JSONDecodeError, ValueError) as e:
            return {"success": False, "error": f"Gemini Hallucination - Invalid JSON Payload generated: {str(e)}", "raw": raw_text}
        except KeyError as e:
             return {"success": False, "error": f"Vertex API Response Schema changed: {str(e)}"}
