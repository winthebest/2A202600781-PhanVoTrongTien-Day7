"""Demo LLM cho lab — trả lời dựa trên context đã retrieve (không cần API key)."""

from __future__ import annotations

import re


def _extract_parts(prompt: str) -> tuple[str, str]:
    if "Context:" not in prompt or "Question:" not in prompt:
        return "", ""
    before_q, question_block = prompt.split("Question:", 1)
    context = before_q.split("Context:", 1)[-1].strip()
    question = question_block.split("Answer:", 1)[0].strip()
    return context, question


def _find_price_lines(context: str) -> list[str]:
    lines = []
    for line in context.splitlines():
        text = line.strip()
        if not text:
            continue
        lower = text.lower()
        if "giá hiện tại" in lower or "giá gốc" in lower:
            lines.append(text)
    return lines


def _find_room_names(context: str) -> list[str]:
    rooms = []
    patterns = [
        r"PHÒNG\s+([^\n(]+)",
        r"Phòng\s+([^\n(]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, context, flags=re.IGNORECASE):
            name = match.group(1).strip()
            if name and name not in rooms:
                rooms.append(name)
    return rooms[:10]


def context_grounded_llm(prompt: str) -> str:
    """
    Đọc context + question từ prompt RAG, trả lời ngắn gọn chỉ dựa trên context.
    Dùng cho demo/benchmark khi chưa có OpenAI API.
    """
    context, question = _extract_parts(prompt)
    if not context or context == "No relevant context found.":
        return "Tôi không có đủ thông tin trong tài liệu để trả lời câu hỏi này."

    q_lower = question.lower()

    if "giá" in q_lower and ("vé" in q_lower or "bao nhiêu" in q_lower):
        prices = _find_price_lines(context)
        if prices:
            return "Theo tài liệu đã retrieve: " + "; ".join(prices[:2])
        return (
            "Context đã retrieve không chứa dòng giá rõ ràng cho sản phẩm được hỏi. "
            "Không thể xác nhận giá VinWonders Nha Trang từ chunk hiện tại."
        )

    if "phòng" in q_lower and ("xông hơi" in q_lower or "trị liệu" in q_lower):
        spa_rooms = [
            "Băng tuyết", "Gỗ bách", "Sương mây", "Đá muối", "Bulgama", "Than củi", "Hoàng thổ"
        ]
        found = [name for name in spa_rooms if name.lower() in context.lower()]
        if found:
            return "Các phòng trị liệu Aquafield trong context: " + ", ".join(found) + "."
        rooms = _find_room_names(context)
        if rooms and "aquafield" in context.lower():
            return "Các phòng được đề cập trong context: " + ", ".join(rooms) + "."
        return (
            "Context nhắc đến khu trị liệu nhưng chunk đã retrieve không liệt kê đủ "
            "tên 7 phòng xông hơi Aquafield."
        )

    if "bao gồm" in q_lower or "dịch vụ gì" in q_lower:
        if "## Bao gồm" in context or "Các dịch vụ bao gồm" in context:
            excerpt = context[context.find("Bao gồm") : context.find("Bao gồm") + 500]
            items = [line.strip("- ").strip() for line in excerpt.splitlines() if line.strip().startswith("-")]
            if items:
                return "Combo bao gồm: " + "; ".join(items[:4])
        return "Context chưa có mục 'Bao gồm' rõ ràng; chỉ thấy mô tả chung về khách sạn."

    if "sunrise" in q_lower or "tee-time" in q_lower or "tee time" in q_lower:
        if "Sunrise" in context or "tee-time" in context.lower():
            details: list[str] = []
            capture = False
            for line in context.splitlines():
                text = line.strip()
                if not text:
                    continue
                if text.startswith("Sunrise") or text == "Sunrise":
                    capture = True
                elif capture and text[0].isupper() and "tee" not in text.lower() and "thứ" not in text:
                    if not any(k in text.lower() for k in ("tee", "áp dụng", "khách", "lưu ý", "thời hạn")):
                        break
                if capture:
                    details.append(text)
                if capture and len(details) >= 6:
                    break
            if details:
                return "Theo điều khoản voucher golf (Sunrise): " + " ".join(details)[:450]
            snippet = context[:500].replace("\n", " ")
            return "Theo điều khoản voucher golf: " + snippet
        return "Context không chứa thông tin Sunrise tee-time."

    if "night safari" in q_lower or "safari" in q_lower:
        if "Night Safari" in context or "ban đêm" in context.lower():
            for sentence in re.split(r"(?<=[.!?])\s+", context.replace("\n", " ")):
                if "night safari" in sentence.lower() or "ban đêm" in sentence.lower():
                    return sentence.strip()[:350]
        return (
            "Context đã retrieve không chứa mô tả Night Safari Vinpearl Safari Phú Quốc. "
            "Agent không thể trả lời đúng từ chunk hiện tại."
        )

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context.replace("\n", " ")) if s.strip()]
    if sentences:
        return "Dựa trên context: " + " ".join(sentences[:2])[:400]
    return "Tôi không có đủ thông tin trong tài liệu để trả lời."
