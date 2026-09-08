from google import genai

client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input="You are the AI intelligence layer for SafeGuardOS. Reply with exactly: SafeGuardOS Gemini connection successful."
)

print(interaction.output_text)
