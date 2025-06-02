# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import mimetypes
import os
import re
import struct
from google import genai
from google.genai import types


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to: {file_name}")


def generate():
    # Check if API key is set
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set!")
        print("Please set it with: export GEMINI_API_KEY='your_api_key_here'")
        return

    print("Starting audio generation...")
    print(f"API key found: {'*' * 10}{api_key[-4:]}")  # Show last 4 chars for verification
    
    client = genai.Client(api_key=api_key)

    model = "gemini-2.5-flash-preview-tts"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""Read aloud in a realistic comfortable tone:
Speaker 1: Tsena.

Speaker 2: Dumela, rra.

Speaker 1: Dumela, mma. Tsaya setulo.

Speaker 2: Ke a leboga, rra.

Speaker 1: O tsogile jang, mma?

Speaker 2: Ga ke a tsoga sentle, rra. Ke a lwala.

Speaker 1: Ke a bona. Mathata ke eng?

Speaker 2: Ke na le mogote, rra. Mme ke a gotlhola, bogolo thata bosigo. Gape mmele otlhe o utlwa botlhoko, o ka re o a kgaoga.

Speaker 1: Ke a utlwa. Fa o gotlhola, a o ntsha mogobo?

Speaker 2: Ee, rra. Go na le sengwe se se tswang. Se mmala o mosetlha.

Speaker 1: Mme mogote, o na le ona nako e kae?

Speaker 2: E simolotse maloba, rra. E ya godimo le kwa tlase.

Speaker 1: A o na le mathata a mangwe? A o na le letshoroma kgotsa o a feroga sebete?

Speaker 2: Ee, rra, letshoroma le a ntlhasela. Ga ke feroge sebete, mme ga ke na keletso ya dijo.

Speaker 1: Ke a go utlwa. Jaanong ke tla go lekola mme. Tsweetswee, hema o ntsha moya mo go nna.

Speaker 2: Go botlhoko fa ke hema thata. Sefuba se a omella.

Speaker 1: Go siame, mma. Go ya ka seo o se mpolelelang le tlhahlobo, go lelega jaaka o na le bolwetse jwa sefuba. Bolwetse jono bo a tlwaelega, segolo jang mo nakong eno.

Speaker 2: A go masisi, rra?

Speaker 1: O se ka wa tshwenyega, re tla go tlhokomela. Ke tla go fa ditharabololo tsa dikokoana-hloko go lwantsha bolwetse. O tshwanetse go tsaya lengolo lotlhe, le fa o simolola go ikutlwa o le botoka. Gape ke tlaa go fa selo se se go alalang go gotlholola le dipilisi tsa mogote le botlhoko jwa mmele.

Speaker 2: Ke a leboga, rra.

Speaker 1: Go botlhokwa gore o ipomile sentle mme o nwe metsi a lekaneng – metsi le tee, mma. Leka go ja, le fa o sena keletso, gore o tshelele maatla.

Speaker 2: Go siame, rra. Ke tla leka.

Speaker 1: O tlaa di bona kwa phaphadisong ya ditlhare. Fa mogote o sa tsube mo malatsing a mararo a a tlang, kgotsa fa o ikutlwa o isiwa ke bolwetse, o tshwanetse go boa gape. A go siame?

Speaker 2: Ee, rra. Ke utlwile.

Speaker 1: Sala sentle, mma.

Speaker 2: Tsamaya sentle, rra."""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=[
            "audio",
        ],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 1",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Zephyr"
                            )
                        ),
                    ),
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 2",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Puck"
                            )
                        ),
                    ),
                ]
            ),
        ),
    )

    print("Sending request to Gemini API...")
    
    try:
        file_index = 0
        chunk_count = 0
        
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            chunk_count += 1
            print(f"Received chunk {chunk_count}")
            
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                print(f"Chunk {chunk_count}: No content")
                continue
                
            if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
                file_name = f"setswana_medical_dialogue_part_{file_index}"
                file_index += 1
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                data_buffer = inline_data.data
                print(f"Received audio data: {len(data_buffer)} bytes, MIME type: {inline_data.mime_type}")
                
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                if file_extension is None:
                    file_extension = ".wav"
                    data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)
                save_binary_file(f"{file_name}{file_extension}", data_buffer)
            else:
                if hasattr(chunk, 'text') and chunk.text:
                    print(f"Text chunk: {chunk.text}")
                else:
                    print(f"Chunk {chunk_count}: No audio or text data")

        if file_index == 0:
            print("No audio files were generated. This could be because:")
            print("1. The API key is invalid")
            print("2. The model doesn't support audio generation")
            print("3. There was an error in the request")
        else:
            print(f"Successfully generated {file_index} audio file(s)")
            
    except Exception as e:
        print(f"Error during API call: {e}")

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given audio data and parameters.

    Args:
        audio_data: The raw audio data as a bytes object.
        mime_type: Mime type of the audio data.

    Returns:
        A bytes object representing the WAV file header.
    """
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size

    # http://soundfile.sapp.org/doc/WaveFormat/

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",          # ChunkID
        chunk_size,       # ChunkSize (total file size - 8 bytes)
        b"WAVE",          # Format
        b"fmt ",          # Subchunk1ID
        16,               # Subchunk1Size (16 for PCM)
        1,                # AudioFormat (1 for PCM)
        num_channels,     # NumChannels
        sample_rate,      # SampleRate
        byte_rate,        # ByteRate
        block_align,      # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",          # Subchunk2ID
        data_size         # Subchunk2Size (size of audio data)
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Parses bits per sample and rate from an audio MIME type string.

    Assumes bits per sample is encoded like "L16" and rate as "rate=xxxxx".

    Args:
        mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

    Returns:
        A dictionary with "bits_per_sample" and "rate" keys. Values will be
        integers if found, otherwise None.
    """
    bits_per_sample = 16
    rate = 24000

    # Extract rate from parameters
    parts = mime_type.split(";")
    for param in parts: # Skip the main type part
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                # Handle cases like "rate=" with no value or non-integer value
                pass # Keep rate as default
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass # Keep bits_per_sample as default if conversion fails

    return {"bits_per_sample": bits_per_sample, "rate": rate}


if __name__ == "__main__":
    generate()
