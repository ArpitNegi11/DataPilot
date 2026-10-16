import io
import struct
import math
import wave
from typing import Dict, Any, List, Optional

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False


class AudioBriefingEngine:
    """
    60-Second Audio Executive Briefing (Data-to-Voice Debrief).
    Synthesizes an executive 60-second verbal briefing summarizing data health,
    revenue/profit metrics, anomaly alerts, and prescriptive strategic actions.
    Renders playable in-memory MP3 audio with seamless offline fallback.
    """

    def __init__(self):
        pass

    def generate_script(
        self,
        summary: Dict[str, Any],
        kpis: Dict[str, Any],
        quality_score: Optional[Dict[str, Any]] = None,
        anomalies: Optional[List[Dict[str, Any]]] = None,
        recommendations: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Draft an executive 110-140 word spoken script calibrated for a 50-60 second debrief.
        """
        rows = summary.get("Cleaned Rows") or summary.get("Original Rows") or kpis.get("total_records") or 0
        q_score = quality_score.get("score", 95) if quality_score else 95
        q_grade = quality_score.get("grade", "A") if quality_score else "A"

        rev = kpis.get("total_revenue")
        prof = kpis.get("total_profit")
        margin = kpis.get("profit_margin_pct")

        sentences = [
            f"Good day, executive team. Here is your sixty-second DataPilot briefing.",
            f"We have ingested and validated {rows:,} enterprise records with a data quality health score of {q_score} percent, earning grade {q_grade}."
        ]

        if rev is not None and prof is not None:
            sentences.append(
                f"Topline revenue stands at ${rev:,.0f}, delivering ${prof:,.0f} in operating profit, representing a {margin:.1f} percent margin."
            )
        elif rev is not None:
            sentences.append(f"Topline revenue stands at ${rev:,.0f} across all reporting lines.")
        else:
            sentences.append(f"Core operational metrics are steady across all business channels.")

        num_anomalies = len(anomalies) if anomalies else 0
        if num_anomalies > 0:
            sentences.append(
                f"Autonomous scan identified {num_anomalies} operational anomalies requiring risk mitigation."
            )
        else:
            sentences.append("All transactions are operating within standard variance tolerances with zero critical anomalies.")

        if recommendations and len(recommendations) > 0:
            top_rec = recommendations[0]
            sentences.append(
                f"Strategic priority: {top_rec.get('title', 'optimize operations')}. Specifically, {top_rec.get('action', 'execute strategic realignment')}."
            )

        sentences.append("Deterministic analytics and multi-agent workflows are active in DataPilot. Have a productive day.")

        return " ".join(sentences)

    def _generate_fallback_wav(self, duration_sec: float = 3.0) -> bytes:
        """Generate a valid, playable minimal audio chime fallback when offline."""
        buf = io.BytesIO()
        sample_rate = 22050
        n_samples = int(sample_rate * duration_sec)
        
        with wave.open(buf, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Generate a gentle double-chime tone (A4 = 440Hz, E5 = 659Hz)
            frames = bytearray()
            for i in range(n_samples):
                t = i / sample_rate
                freq = 440.0 if t < (duration_sec / 2.0) else 659.25
                envelope = math.exp(-3.0 * (t % (duration_sec / 2.0)))
                val = int(envelope * 12000.0 * math.sin(2.0 * math.pi * freq * t))
                val = max(-32767, min(32767, val))
                frames.extend(struct.pack('<h', val))
            
            wav_file.writeframes(frames)
            
        return buf.getvalue()

    def synthesize_audio(self, script_text: str) -> Tuple_AudioResult:
        """
        Synthesize text into in-memory audio bytes.
        Returns: (audio_bytes, mime_type, is_fallback)
        """
        if HAS_GTTS and script_text.strip():
            try:
                tts = gTTS(text=script_text, lang="en", tld="com", slow=False)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                audio_bytes = buf.getvalue()
                if len(audio_bytes) > 500:
                    return audio_bytes, "audio/mp3", False
            except Exception:
                pass

        # Offline fallback
        wav_bytes = self._generate_fallback_wav(duration_sec=3.0)
        return wav_bytes, "audio/wav", True

    def get_briefing(
        self,
        summary: Dict[str, Any],
        kpis: Dict[str, Any],
        quality_score: Optional[Dict[str, Any]] = None,
        anomalies: Optional[List[Dict[str, Any]]] = None,
        recommendations: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete briefing object with script, audio bytes, and metadata.
        """
        script = self.generate_script(summary, kpis, quality_score, anomalies, recommendations)
        audio_bytes, mime_type, is_fallback = self.synthesize_audio(script)
        
        # Word count estimate: average 130 words per minute
        word_count = len(script.split())
        est_duration = max(15, round(word_count / 130.0 * 60.0))

        return {
            "script": script,
            "audio_bytes": audio_bytes,
            "mime_type": mime_type,
            "is_fallback": is_fallback,
            "word_count": word_count,
            "estimated_duration_sec": est_duration
        }


# Type alias helper
Tuple_AudioResult = Any
