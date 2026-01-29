# Build an AI Image Generator app using Imagen on Vertex AI

## What This Lab Is About

### Understanding the Title

**Build an AI Image Generator app using Imagen on Vertex AI**

This title describes exactly what we are going to do in this lab:

- **Build**: We will write and run a small but complete Python program.
- **AI Image Generator app**: The program will generate an image using Artificial Intelligence based on a text description.
- **Imagen**: Imagen is Google’s state-of-the-art text-to-image generative AI model.
- **Vertex AI**: Vertex AI is Google Cloud’s managed platform where Imagen is hosted and accessed.

In simple terms, during this lab we will:

1. Connect our Python code to Vertex AI
2. Use a pre-trained Imagen model (no training required)
3. Send a text prompt (for example, a description of a scene)
4. Receive an AI-generated image as output

This lab demonstrates how developers can quickly add **Generative AI capabilities** to an application without managing infrastructure or building models from scratch.

---

## Overview

This lab demonstrates how to generate an image from a text prompt using **Imagen on Vertex AI**. You will use the Vertex AI Python SDK to load a pre-trained image generation model, send a prompt, and save the generated image locally.

Tip: The FAQ with 20+ questions is at the end of this document.

---

![AI Image Generator App Architecture](https://github.com/data-abhishek/AWS-Certified-Data-Analyst-Preparation/blob/main/chromalabimage.png)

---

## Source Code (Main File)

```python
import argparse

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

def generate_image(
    project_id: str, location: str, output_file: str, prompt: str
) -> vertexai.preview.vision_models.ImageGenerationResponse:
    """Generate an image using a text prompt.
    Args:
      project_id: Google Cloud project ID, used to initialize Vertex AI.
      location: Google Cloud region, used to initialize Vertex AI.
      output_file: Local path to the output image file.
      prompt: The text prompt describing what you want to see."""

    vertexai.init(project=project_id, location=location)

    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

    images = model.generate_images(
        prompt=prompt,
        # Optional parameters
        number_of_images=1,
        seed=1,
        add_watermark=False,
    )

    images[0].save(location=output_file)

    return images

generate_image(
    project_id='qwiklabs-gcp-02-c10e2028f324',
    location='europe-west1',
    output_file='image.jpeg',
    prompt='Create an image of a cricket ground in the heart of Los Angeles',
)
```

---

## Code Explanation

### Import Statements

```python
import argparse
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
```

- `argparse`: Commonly used for command-line arguments. Included for extensibility but not used in this script.
- `vertexai`: Initializes and communicates with Vertex AI services.
- `ImageGenerationModel`: Provides access to Imagen text-to-image models.

---

### Function Definition

```python
def generate_image(
    project_id: str, location: str, output_file: str, prompt: str
)
```

- Defines a reusable function to generate images.
- Parameters:
  - `project_id`: Google Cloud project identifier.
  - `location`: Region where Vertex AI runs.
  - `output_file`: File name where the image will be saved.
  - `prompt`: Text description used to generate the image.

---

### Initialize Vertex AI

```python
vertexai.init(project=project_id, location=location)
```

- Connects the script to Vertex AI.
- Required before calling any Vertex AI models.

---

### Load the Imagen Model

```python
model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
```

- Loads a pre-trained Imagen image generation model.
- No model training or deployment is required.

---

### Generate the Image

```python
images = model.generate_images(
    prompt=prompt,
    number_of_images=1,
    seed=1,
    add_watermark=False,
)
```

- Sends the text prompt to the model.
- `number_of_images=1`: Generates a single image.
- `seed=1`: Makes the output reproducible.
- `add_watermark=False`: Disables the default SynthID watermark.

---

### Save the Image

```python
images[0].save(location=output_file)
```

- Saves the generated image locally using the provided file name.

---

### Function Call

```python
generate_image(
    project_id='qwiklabs-gcp-02-c10e2028f324',
    location='europe-west1',
    output_file='image.jpeg',
    prompt='Create an image of a cricket ground in the heart of Los Angeles',
)
```

- Executes the function.
- Generates an image based on the prompt and saves it as `image.jpeg`.

---

## Output

After running the script, an image file named **image.jpeg** is created in the working directory, containing the AI-generated image based on the given prompt.

---

## FAQ (20+ Questions)

Below, each question is presented as a clear subheading, followed by its answer for easy scanning.

### Q1. What does this script do at a high level?
Answer: It initializes Vertex AI, loads an Imagen 3 model, generates an image from your text prompt, and saves the first image to a file.

### Q2. Which Imagen model is used here?
Answer: `ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")`, a Vertex AI–hosted Imagen 3 text-to-image model.

### Q3. Where do I set my Google Cloud project and region?
Answer: In the `generate_image` call (`project_id`, `location`) which are passed to `vertexai.init()`.

### Q4. How do I change the output file path or format?
Answer: Modify `output_file` (e.g., `result.png` or a full path like `C:/tmp/result.jpeg`).

### Q5. How can I generate multiple images?
Answer: Increase `number_of_images` and then iterate over the returned list: `for i, img in enumerate(images): img.save(location=f"image_{i+1}.jpeg")`.

### Q6. What does the `seed` parameter do?
Answer: It controls reproducibility. Using the same prompt and seed tends to produce similar outputs. Vary or remove for diversity.

### Q7. Why is `add_watermark=False` used?
Answer: To disable SynthID watermark embedding. Ensure this aligns with your compliance requirements.

### Q8. How can I pass prompts dynamically from the command line?
Answer: Use `argparse` to accept a `--prompt` argument and pass it to `generate_image(prompt=args.prompt, ...)`.

### Q9. Do I need to train or deploy a model on Vertex AI?
Answer: No. This uses a pre-trained hosted model—no training or deployment is required.

### Q10. What IAM permissions are needed?
Answer: Typically `roles/aiplatform.user` and basic project access. Authenticate via Application Default Credentials (ADC) or a service account.

### Q11. How do I run this locally end-to-end?
Answer: `pip install google-cloud-aiplatform`; `gcloud auth application-default login`; run the Python script.

### Q12. Can I control image size or aspect ratio?
Answer: Check the Imagen SDK docs for supported kwargs (e.g., width/height) for your model version and region; pass them in `generate_images()` if available.

### Q13. How do I influence style (e.g., photorealistic, watercolor, 3D render)?
Answer: Describe the style directly in your prompt; some versions may provide additional control knobs—see the docs.

### Q14. How should I handle errors, quotas, and region support issues?
Answer: Wrap API calls in try/except, log exceptions, verify IAM roles, enable billing, check regional availability, and monitor quotas.

### Q15. What’s a good way to name multiple outputs uniquely?
Answer: Use indices or timestamps, e.g., `image_{i+1}.jpeg` or `image_{int(time.time())}.png`.

### Q16. Is `argparse` required in this minimal example?
Answer: Not required, but recommended to parameterize project, region, prompt, and output file for scripting and automation.

### Q17. Which regions typically support Imagen 3?
Answer: Refer to Vertex AI docs; commonly `us-central1` and `europe-west1` support Imagen models (availability can change).

### Q18. Can I integrate this into a web application?
Answer: Yes—expose an API endpoint that accepts a prompt, calls `generate_image`, stores the file (e.g., Cloud Storage), and returns a URL.

### Q19. How do I manage costs?
Answer: Limit `number_of_images`, validate/canonicalize prompts, cache repeated prompts, and monitor usage/quota in Cloud Console.

### Q20. How can I make outputs deterministic for tests?
Answer: Pin the SDK/model versions, fix the `seed`, and keep prompt/parameters constant.

### Q21. What authentication methods can I use locally?
Answer: `gcloud auth application-default login` (ADC) or a service account JSON key via `GOOGLE_APPLICATION_CREDENTIALS` env var.

### Q22. How do I check or increase quotas?
Answer: Visit Vertex AI quotas in Cloud Console; request increases if needed.

### Q23. How do I enable more verbose logging for debugging?
Answer: Configure Python logging (e.g., `logging.basicConfig(level=logging.INFO)`) and log prompts/parameters carefully (avoid secrets).

### Q24. Can I package this as a CLI tool?
Answer: Yes—add `argparse` flags for project, region, prompt, seed, number_of_images, output path, and wrap main logic in `if __name__ == "__main__":`.

