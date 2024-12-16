import random
from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

from src.config import RANDOM_STATE, RAW_DATA_DIR

app = typer.Typer()

# Set the random state for reproducibility
random.seed(RANDOM_STATE)

# Define the image size and shape parameters
IMAGE_SIZE = (224, 224)
KNOWN_SHAPES = ['circle', 'square', 'triangle']


def generate_random_grey_shade():
    """Generate a random shade of grey."""
    shade = random.randint(0, 255)
    return (shade, shade, shade)


def find_non_overlapping_position(size, existing_shapes):
    """Find a non-overlapping position for a shape."""
    max_attempts = 1000

    for _ in range(max_attempts):
        x = random.randint(0, IMAGE_SIZE[0] - size)
        y = random.randint(0, IMAGE_SIZE[1] - size)

        new_shape = (x, y, x + size, y + size)

        if not any(is_overlap(new_shape, shape) for shape in existing_shapes):
            return new_shape

    return None


def is_overlap(new_shape, existing_shape):
    """Check if the new shape overlaps with an existing shape."""
    x1, y1, x2, y2 = new_shape
    ex1, ey1, ex2, ey2 = existing_shape
    return not (x2 <= ex1 or x1 >= ex2 or y2 <= ey1 or y1 >= ey2)


def draw_shape(draw, shape_type, size, existing_shapes, flags):
    """Draw the specified shape on the image ensuring no overlap."""
    position = find_non_overlapping_position(size, existing_shapes)
    if position is None:
        logger.warning("Could not place shape without overlap. Skipping shape.")
        return

    x1, y1, x2, y2 = position
    color = generate_random_grey_shade()

    if shape_type == 'circle':
        draw.ellipse(position, fill=color, outline=color)
        flags['circle'] = 1
    elif shape_type == 'square':
        draw.rectangle(position, fill=color, outline=color)
        flags['square'] = 1
    elif shape_type == 'triangle':
        draw.polygon([(x1 + size // 2, y1), (x2, y2), (x1, y2)], fill=color, outline=color)
        flags['triangle'] = 1

    existing_shapes.append(position)


def generate_random_blot(image):
    """Generate a random blot-like shape."""
    blot = Image.new('L', (IMAGE_SIZE[0] // 4, IMAGE_SIZE[1] // 4), 0)
    blot_draw = ImageDraw.Draw(blot)

    # Randomly draw some overlapping circles to create a blot effect
    for _ in range(random.randint(5, 10)):
        x = random.randint(0, blot.size[0])
        y = random.randint(0, blot.size[1])
        radius = random.randint(10, 50)
        blot_draw.ellipse([x, y, x + radius, y + radius], fill=255)

    # Apply a blur to the blot to soften edges
    blot = blot.filter(ImageFilter.GaussianBlur(radius=5))

    # Randomly place the blot on the original image
    x = random.randint(0, IMAGE_SIZE[0] - blot.size[0])
    y = random.randint(0, IMAGE_SIZE[1] - blot.size[1])

    # Convert blot to grayscale and paste onto the original image
    image.paste(blot.convert('RGB'), (x, y), blot)


def draw_random_lines(draw, num_lines=5):
    """Draw random lines on the image."""
    for _ in range(num_lines):
        start_pos = (random.randint(0, IMAGE_SIZE[0]), random.randint(0, IMAGE_SIZE[1]))
        end_pos = (random.randint(0, IMAGE_SIZE[0]), random.randint(0, IMAGE_SIZE[1]))
        color = generate_random_grey_shade()
        thickness = random.randint(1, 5)
        draw.line([start_pos, end_pos], fill=color, width=thickness)


@app.command()
def generate_images(train_output_dir: Path = RAW_DATA_DIR / 'synthetic' / 'train',
                    test_output_dir: Path = RAW_DATA_DIR / 'synthetic' / 'test',
                    num_train_images: int = 2500,
                    num_test_images: int = 500,
                    test_mode: str = 'lines'):
    """Generate images with random geometric shapes and add out-of-domain artifacts to test images."""
    train_output_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('Generating train images')

    # List to store flags for each train image
    train_flags_list = []

    # Generate train images
    for i in tqdm(range(num_train_images), desc='Generating train images', unit='image'):
        image = Image.new('RGB', IMAGE_SIZE, 'white')
        draw = ImageDraw.Draw(image)

        existing_shapes = []  # Keep track of placed shapes to avoid overlap

        # Initialize flags for each shape
        flags = {'circle': 0, 'square': 0, 'triangle': 0, 'image_name': f'train_image_{i + 1}.png'}

        # Decide how many shapes to draw
        num_shapes = random.randint(1, 5)

        for _ in range(num_shapes):
            shape_type = random.choice(KNOWN_SHAPES)
            size = random.randint(30, 80)
            draw_shape(draw, shape_type, size, existing_shapes, flags)

        # Save the image
        image_path = train_output_dir / flags['image_name']
        image.save(image_path)

        # Append the flags to the train list
        train_flags_list.append(flags)

    # Save train flags to a parquet file
    train_flags_df = pd.DataFrame(train_flags_list)
    train_flags_df.to_parquet(train_output_dir / 'shape_flags.parquet', index=False)

    logger.info('Generating test images')

    # List to store flags for each test image
    test_flags_list = []

    # Generate test images
    for i in tqdm(range(num_test_images), desc='Generating test images', unit='image'):
        image = Image.new('RGB', IMAGE_SIZE, 'white')

        # Step 1: Add either blot or lines to the image based on the test_mode
        if test_mode == 'blot':
            generate_random_blot(image)
            image = image.filter(ImageFilter.GaussianBlur(radius=5))

        # Step 2: Add geometric shapes on top of the blurred image
        draw = ImageDraw.Draw(image)
        existing_shapes = []  # Keep track of placed shapes to avoid overlap

        # Initialize flags for test images
        flags = {'circle': 0, 'square': 0, 'triangle': 0, 'image_name': f'test_image_{i + 1}.png'}

        # Decide how many shapes to draw after the image is blurred
        num_shapes = random.randint(1, 5)

        for _ in range(num_shapes):
            shape_type = random.choice(KNOWN_SHAPES)
            size = random.randint(30, 80)
            draw_shape(draw, shape_type, size, existing_shapes, flags)

        # Step 3: Draw lines on top of the shapes if in 'lines' mode
        if test_mode == 'lines':
            draw_random_lines(draw)

        # Save the test image
        image_path = test_output_dir / flags['image_name']
        image.save(image_path)

        # Append test image flags to the list
        test_flags_list.append(flags)

    # Save test flags to a parquet file
    test_flags_df = pd.DataFrame(test_flags_list)
    test_flags_df.to_parquet(test_output_dir / 'shape_flags.parquet', index=False)

    logger.success('Image generation complete')


if __name__ == '__main__':
    app()
