from google import genai
from PIL import Image

client = genai.Client()

image = Image.open("/tmp/argus_test_frame.jpg")

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=[
        """
You are the visual intelligence layer of SafeGuardOS.

Analyse this aerial security-camera image.

Describe:
1. How many people are visible.
2. What they appear to be doing.
3. Whether there appears to be any notable interaction between them.
4. Any potentially concerning behaviour.
5. Your confidence and uncertainty.

Do NOT claim that someone is committing a crime.
Do NOT identify people.
Describe observable behaviour and evidence only.
""",
        image,
    ],
)

print(response.text)
