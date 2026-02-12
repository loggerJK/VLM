# -*- coding: utf-8 -*-
"""
GLM-OCR Test Script

Tests GLM-OCR vLLM server on images in a specified directory or a single image.

Usage:
    # Single image
    python evaluation_scripts/glm_ocr_test.py --image_path /path/to/image.png

    # Directory of images
    python evaluation_scripts/glm_ocr_test.py --image_path /path/to/images/

    # Custom server URL
    python evaluation_scripts/glm_ocr_test.py --image_path /path/to/image.png --api_base http://localhost:8090
"""
import os
import sys
import argparse
import base64
import json
import time
import requests


def extract_text_with_glmocr(image_path, api_base="http://localhost:8090"):
    """Extract text from image using GLM-OCR via vLLM server."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/png")

    response = requests.post(
        f"{api_base}/v1/chat/completions",
        json={
            "model": "glm-ocr",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": "OCR this image. Return only the extracted text."},
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.01,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def collect_images(path):
    """Collect image file paths from a file or directory."""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        files = []
        for f in sorted(os.listdir(path)):
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(path, f))
        return files
    else:
        print(f"Error: {path} does not exist.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="GLM-OCR Test")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to a single image or a directory of images")
    parser.add_argument("--api_base", type=str, default="http://localhost:8090",
                        help="GLM-OCR vLLM server URL")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional path to save results as JSON")
    args = parser.parse_args()

    # Check server health
    try:
        r = requests.get(f"{args.api_base}/v1/models", timeout=5)
        r.raise_for_status()
        print(f"Server OK: {args.api_base}")
    except Exception as e:
        print(f"Cannot reach GLM-OCR server at {args.api_base}: {e}")
        sys.exit(1)

    images = collect_images(args.image_path)
    print(f"Found {len(images)} image(s)\n")

    results = []
    for img_path in images:
        print(f"Processing: {img_path}")
        t0 = time.time()
        try:
            text = extract_text_with_glmocr(img_path, api_base=args.api_base)
            elapsed = time.time() - t0
            print(f"  OCR result ({elapsed:.2f}s): {text}")
            results.append({"image": img_path, "ocr_text": text, "time": round(elapsed, 2)})
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  Error ({elapsed:.2f}s): {e}")
            results.append({"image": img_path, "ocr_text": None, "error": str(e), "time": round(elapsed, 2)})
        print()

    # Summary
    success = sum(1 for r in results if r.get("ocr_text") is not None)
    print("=" * 50)
    print(f"Done: {success}/{len(results)} succeeded")
    if results:
        avg_time = sum(r["time"] for r in results) / len(results)
        print(f"Avg time per image: {avg_time:.2f}s")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
