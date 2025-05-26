# %%
import base64
import asyncio
from openai import AsyncOpenAI
import pandas as pd 
from typing import List, Dict, Any
import logging
import json
import time
from pathlib import Path
import os
from dotenv import load_dotenv

# Clear any existing environment variables to force reload
if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']

# Load environment variables from .env file in project root
project_root = Path(__file__).parent.parent  # Go up from openai/ to street-flooding/
env_path = project_root / '.env'
loaded = load_dotenv(dotenv_path=env_path, override=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_environment():
    """Validate that necessary environment variables and dependencies are set up"""
    import os
    
    logger.info(f"Checking for .env file at: {env_path}")
    logger.info(f".env file exists: {env_path.exists()}")
    logger.info(f"dotenv loaded successfully: {loaded}")
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(f"OPENAI_API_KEY environment variable not set. Checked .env file at: {env_path}")
    
    # Only accept sk-proj- keys for project-oriented organizations
    if not api_key.startswith('sk-proj-'):
        raise ValueError(f"OPENAI_API_KEY must start with 'sk-proj-' for project-oriented organizations. Found key starting with: {api_key[:10]}...")
    
    if len(api_key) < 50:  # Project keys are longer
        raise ValueError(f"OPENAI_API_KEY appears to be too short. Expected project key length. Length: {len(api_key)}")
    
    logger.info(f"✅ Environment validation passed (loaded from {env_path})")
    logger.info(f"✅ Using project API key: {api_key[:15]}...")
    return True

# Initialize client after environment validation
client = AsyncOpenAI()

def validate_csv_and_images(csv_path: str, image_column: str = 'image_path', max_check: int = 10):
    """Validate CSV file and check that image paths are valid"""
    
    # Check CSV exists and is readable
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Could not read CSV file: {e}")
    
    if len(df) == 0:
        raise ValueError("CSV file is empty")
    
    # Check if image column exists
    if image_column not in df.columns:
        raise ValueError(f"Column '{image_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Check for missing image paths
    missing_paths = df[image_column].isna().sum()
    if missing_paths == len(df):
        raise ValueError("All image paths are missing/null")
    
    if missing_paths > 0:
        logger.warning(f"Found {missing_paths} rows with missing image paths ({missing_paths/len(df)*100:.1f}%)")
    
    # Sample check that images exist and are readable
    valid_df = df.dropna(subset=[image_column])
    sample_size = min(max_check, len(valid_df))
    sample_paths = valid_df[image_column].sample(sample_size).tolist()
    
    valid_images = 0
    total_size = 0
    
    for path in sample_paths:
        try:
            if not os.path.exists(path):
                logger.warning(f"Image file not found: {path}")
                continue
            
            # Check if file is readable and get size
            size = os.path.getsize(path)
            if size == 0:
                logger.warning(f"Empty image file: {path}")
                continue
            
            # Check if file has valid image extension
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
            if not any(path.lower().endswith(ext) for ext in valid_extensions):
                logger.warning(f"File may not be an image (no valid extension): {path}")
            
            total_size += size
            valid_images += 1
            
        except Exception as e:
            logger.warning(f"Could not access image {path}: {e}")
    
    if valid_images == 0:
        raise ValueError(f"No valid images found in sample of {sample_size} files. Check image paths and file permissions.")
    
    success_rate = valid_images / sample_size * 100
    avg_size_mb = total_size / valid_images / 1024 / 1024
    
    if success_rate < 50:
        raise ValueError(f"Low image validation success rate: {success_rate:.1f}%. Check image paths.")
    
    logger.info(f"✅ Image validation passed: {valid_images}/{sample_size} images valid ({success_rate:.1f}%)")
    logger.info(f"✅ Average image size: {avg_size_mb:.1f}MB")
    
    return {
        'total_rows': len(df),
        'valid_image_paths': len(valid_df),
        'missing_paths': missing_paths,
        'sample_success_rate': success_rate,
        'avg_image_size_mb': avg_size_mb
    }

def validate_model(model: str):
    """Validate that the model name is supported"""
    supported_models = {
        'o1-mini', 'o1-preview', 
        'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4'
    }
    
    if model not in supported_models:
        logger.warning(f"Model '{model}' may not be supported. Supported models: {supported_models}")
        # Don't raise error as new models may be added
    
    logger.info(f"✅ Model validation: {model}")
    return True

def validate_prompt(prompt: str):
    """Validate prompt is reasonable"""
    if not prompt or len(prompt.strip()) == 0:
        raise ValueError("Prompt cannot be empty")
    
    if len(prompt) > 4000:  # Reasonable limit
        raise ValueError(f"Prompt is very long ({len(prompt)} chars). Consider shortening.")
    
    logger.info(f"✅ Prompt validation: {len(prompt)} characters")
    return True

def sample_dataset(df: pd.DataFrame, sample_percentage: float = 10.0, random_seed: int = 42):
    """
    Take a fixed-seed random sample of the dataset
    
    Args:
        df: DataFrame to sample from
        sample_percentage: Percentage of dataset to sample (0-100)
        random_seed: Fixed random seed for reproducible sampling
        
    Returns:
        Sampled DataFrame
    """
    if sample_percentage <= 0 or sample_percentage > 100:
        logger.error(f"Invalid sample percentage: {sample_percentage}. Must be between 0 and 100.")
        return df
    
    if sample_percentage >= 100:
        logger.info("Sample percentage >= 100%, returning full dataset")
        return df
    
    sample_size = int(len(df) * sample_percentage / 100)
    
    if sample_size == 0:
        logger.warning("Sample size calculated as 0, returning single row")
        sample_size = 1
    
    logger.info(f"Sampling {sample_size:,} images ({sample_percentage}%) from {len(df):,} total images with seed {random_seed}")
    
    # Use fixed seed for reproducible sampling
    sampled_df = df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
    
    return sampled_df

def estimate_request_size(image_path: str, prompt: str) -> int:
    """Estimate the size of a single request in bytes"""
    try:
        # Get image file size
        if not os.path.exists(image_path):
            logger.warning(f"Image not found for size estimation: {image_path}")
            return 2 * 1024 * 1024  # 2MB default
        
        image_size = os.path.getsize(image_path)
        
        # Validate image size is reasonable
        max_image_size = 20 * 1024 * 1024  # 20MB
        if image_size > max_image_size:
            logger.warning(f"Large image file ({image_size/1024/1024:.1f}MB): {image_path}")
        
        # Base64 encoding increases size by ~33%
        base64_size = int(image_size * 1.33)
        
        # JSON overhead (approximate)
        json_overhead = len(prompt) + 500  # Rough estimate for JSON structure
        
        total_size = base64_size + json_overhead
        return total_size
    except Exception as e:
        logger.warning(f"Could not estimate size for {image_path}: {e}")
        # If we can't get size, use conservative estimate
        return 2 * 1024 * 1024  # 2MB default

def create_batch_file_size_aware(df: pd.DataFrame, prompt: str, output_path: str = "batch_requests.jsonl", max_size_mb: int = 190, model: str = "gpt-4o-mini"):
    """Create a JSONL file for batch processing with size limits - using 190MB to leave buffer under 200MB limit"""
    
    if len(df) == 0:
        raise ValueError("DataFrame is empty, cannot create batch file")
    
    batch_requests = []
    current_size = 0
    max_size_bytes = max_size_mb * 1024 * 1024
    processed_count = 0
    error_count = 0
    
    logger.info(f"Creating batch file with max size {max_size_mb}MB for {len(df)} images")
    
    for idx, row in df.iterrows():
        # Check if we have image_path column
        if 'image_path' not in row or pd.isna(row['image_path']):
            logger.warning(f"Row {idx}: Missing image path, skipping")
            error_count += 1
            continue
            
        image_path = row['image_path']
        
        # Estimate request size first
        estimated_size = estimate_request_size(image_path, prompt)
        
        # Check if adding this request would exceed size limit
        if current_size + estimated_size > max_size_bytes and len(batch_requests) > 0:
            logger.info(f"Stopping batch at {len(batch_requests)} requests to stay under {max_size_mb}MB limit")
            break
        
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}")
                error_count += 1
                continue
                
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                
                if len(image_data) == 0:
                    logger.warning(f"Empty image file: {image_path}")
                    error_count += 1
                    continue
                
                base64_image = base64.b64encode(image_data).decode("utf-8")
            
            request = {
                "custom_id": f"request-{idx}",
                "method": "POST", 
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                }
            }
            
            # Calculate actual size
            request_json = json.dumps(request)
            actual_size = len(request_json.encode('utf-8'))
            
            # Double check size limit
            if current_size + actual_size > max_size_bytes and len(batch_requests) > 0:
                logger.info(f"Hit size limit at {len(batch_requests)} requests ({current_size/1024/1024:.1f}MB)")
                break
                
            batch_requests.append(request)
            current_size += actual_size
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            error_count += 1
            continue
    
    if len(batch_requests) == 0:
        raise ValueError(f"No valid requests created. Processed {processed_count}, errors: {error_count}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Write to JSONL file
    try:
        with open(output_path, 'w') as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
    except Exception as e:
        raise IOError(f"Failed to write batch file {output_path}: {e}")
    
    final_size_mb = current_size / 1024 / 1024
    logger.info(f"Created batch file with {len(batch_requests)} requests ({final_size_mb:.1f}MB): {output_path}")
    
    if error_count > 0:
        logger.warning(f"Skipped {error_count} images due to errors")
    
    return output_path, len(batch_requests), processed_count

def create_batch_files_size_aware(df: pd.DataFrame, prompt: str, output_dir: str = "batch_files", max_size_mb: int = 190, model: str = "gpt-4o-mini"):
    """Create multiple batch files optimized for your 200MB limit"""
    if len(df) == 0:
        raise ValueError("DataFrame is empty, cannot create batch files")
    
    Path(output_dir).mkdir(exist_ok=True)
    batch_files = []
    
    current_idx = 0
    batch_num = 1
    
    # With your high rate limits, we can be more aggressive with batch sizes
    max_requests_per_batch = 50000  # OpenAI limit
    
    while current_idx < len(df):
        # Take up to 50k rows for this batch
        chunk_end = min(current_idx + max_requests_per_batch, len(df))
        chunk_df = df.iloc[current_idx:chunk_end]
        
        output_path = f"{output_dir}/batch_requests_{batch_num}.jsonl"
        
        try:
            file_path, request_count, processed_count = create_batch_file_size_aware(
                chunk_df, prompt, output_path, max_size_mb, model
            )
        except ValueError as e:
            if "No valid requests created" in str(e) and current_idx > 0:
                # If we can't create any requests from this chunk, we're done
                logger.warning(f"Stopping batch creation at batch {batch_num}: {e}")
                break
            else:
                # Re-raise if this is the first batch or a different error
                raise
        
        batch_files.append({
            'file_path': file_path,
            'request_count': request_count,
            'start_idx': current_idx,
            'end_idx': current_idx + processed_count,
            'batch_num': batch_num
        })
        
        logger.info(f"Batch {batch_num}: {request_count} requests, rows {current_idx} to {current_idx + processed_count - 1}")
        
        current_idx += processed_count
        batch_num += 1
        
        # Safety check - if we didn't process any images, break to avoid infinite loop
        if processed_count == 0:
            logger.error(f"Could not process any images starting at index {current_idx}. Images may be too large.")
            break
    
    if len(batch_files) == 0:
        raise ValueError("Could not create any valid batch files. Check your data and image paths.")
    
    logger.info(f"Created {len(batch_files)} batch files for {len(df)} total images using model {model}")
    return batch_files

async def submit_batch_optimized(file_path: str, metadata: dict = None, delay: float = 0.1):
    """Submit a batch file with minimal delay given your high rate limits"""
    # Upload the file
    with open(file_path, "rb") as f:
        file_response = await client.files.create(
            file=f,
            purpose="batch"
        )
    
    # Create the batch
    batch_response = await client.batches.create(
        input_file_id=file_response.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata or {}
    )
    
    logger.info(f"Submitted batch {batch_response.id} for file {file_path}")
    
    # Minimal delay given your high limits
    if delay > 0:
        await asyncio.sleep(delay)
    
    return batch_response

async def download_batch_results(batch_id: str, output_path: str = None):
    """Download and parse batch results"""
    batch = await client.batches.retrieve(batch_id)
    
    if batch.status != "completed":
        logger.error(f"Batch {batch_id} is not completed. Status: {batch.status}")
        return None
    
    # Download output file
    output_file_response = await client.files.content(batch.output_file_id)
    
    if output_path is None:
        output_path = f"batch_results_{batch_id}.jsonl"
    
    with open(output_path, 'wb') as f:
        f.write(output_file_response.content)
    
    logger.info(f"Downloaded batch results to {output_path}")
    return output_path

def parse_batch_results(results_file: str):
    """Parse batch results and return as DataFrame"""
    results = []
    
    with open(results_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            if result.get('response'):
                custom_id = result['custom_id']
                content = result['response']['body']['choices'][0]['message']['content']
                results.append({
                    'custom_id': custom_id,
                    'response': content,
                    'index': int(custom_id.split('-')[1])  # Extract index from custom_id
                })
            else:
                # Handle errors
                custom_id = result['custom_id']
                error = result.get('error', {}).get('message', 'Unknown error')
                results.append({
                    'custom_id': custom_id,
                    'response': f"ERROR: {error}",
                    'index': int(custom_id.split('-')[1])
                })
    
    df_results = pd.DataFrame(results).sort_values('index')
    return df_results

async def process_large_dataset_batch(csv_path: str, prompt: str, output_dir: str = "batch_processing", model: str = "gpt-4o-mini"):
    """Process a very large dataset using batch API with 200MB file limit"""
    Path(output_dir).mkdir(exist_ok=True)
    Path(f"{output_dir}/input").mkdir(exist_ok=True)
    Path(f"{output_dir}/results").mkdir(exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_path)
    logger.info(f"Processing {len(df)} images using Batch API with model {model}")
    logger.info(f"Using 200MB batch file limit (190MB with buffer)")
    
    # Create batch files with 200MB limit (190MB with buffer)
    batch_files = create_batch_files_size_aware(
        df, prompt, 
        output_dir=f"{output_dir}/input", 
        max_size_mb=190,  # Use 190MB to leave 10MB buffer under 200MB limit
        model=model
    )
    
    # Submit all batches - files are uploaded automatically in submit_batch_optimized
    batch_jobs = []
    for batch_info in batch_files:
        metadata = {
            "batch_number": str(batch_info['batch_num']),
            "total_batches": str(len(batch_files)),
            "description": f"Flood detection batch {batch_info['batch_num']}/{len(batch_files)}",
            "model": model,
            "file_size_mb": f"{os.path.getsize(batch_info['file_path'])/1024/1024:.1f}MB"
        }
        
        logger.info(f"Uploading and submitting batch {batch_info['batch_num']} ({metadata['file_size_mb']})")
        batch_response = await submit_batch_optimized(batch_info['file_path'], metadata, delay=0.1)
        batch_jobs.append({
            'batch_id': batch_response.id,
            'batch_info': batch_info,
            'start_idx': batch_info['start_idx'],
            'end_idx': batch_info['end_idx']
        })
        
    logger.info(f"Successfully uploaded and submitted {len(batch_jobs)} batches")
    logger.info("All batch files have been uploaded to OpenAI and processing has started")
    
    return batch_jobs

def analyze_dataset_for_batching(csv_path: str, sample_size: int = 100):
    """Analyze dataset to estimate batch requirements with 200MB limit"""
    
    # Validate inputs
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if sample_size <= 0:
        raise ValueError("Sample size must be positive")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Could not read CSV file: {e}")
    
    if len(df) == 0:
        raise ValueError("CSV file is empty")
    
    if 'image_path' not in df.columns:
        raise ValueError("CSV must have 'image_path' column")
    
    # Sample some images to estimate sizes
    valid_df = df.dropna(subset=['image_path'])
    if len(valid_df) == 0:
        raise ValueError("No valid image paths found in CSV")
    
    sample_df = valid_df.sample(min(sample_size, len(valid_df)))
    
    total_size = 0
    valid_images = 0
    
    for _, row in sample_df.iterrows():
        try:
            if not os.path.exists(row['image_path']):
                continue
            size = os.path.getsize(row['image_path'])
            if size > 0:  # Skip empty files
                total_size += size
                valid_images += 1
        except Exception:
            continue
    
    if valid_images == 0:
        raise ValueError("No valid images found in sample. Check image paths and file permissions.")
    
    avg_image_size = total_size / valid_images
    avg_base64_size = avg_image_size * 1.33  # Base64 overhead
    avg_request_size = avg_base64_size + 500  # JSON overhead
    
    # Estimate batching with 200MB limit
    max_batch_size_mb = 190  # Leave 10MB buffer
    max_batch_size_bytes = max_batch_size_mb * 1024 * 1024
    
    estimated_requests_per_batch = max(1, int(max_batch_size_bytes / avg_request_size))
    estimated_batches = max(1, int(len(valid_df) / estimated_requests_per_batch) + 1)
    
    logger.info(f"Dataset Analysis (200MB batch limit):")
    logger.info(f"  Total images: {len(df):,}")
    logger.info(f"  Valid images: {len(valid_df):,}")
    logger.info(f"  Average image size: {avg_image_size/1024/1024:.1f}MB")
    logger.info(f"  Average request size: {avg_request_size/1024/1024:.1f}MB")
    logger.info(f"  Estimated requests per batch (190MB): {estimated_requests_per_batch:,}")
    logger.info(f"  Estimated total batches: {estimated_batches:.0f}")
    
    return {
        'total_images': len(df),
        'valid_images': len(valid_df),
        'avg_image_size_mb': avg_image_size/1024/1024,
        'estimated_requests_per_batch': estimated_requests_per_batch,
        'estimated_batches': estimated_batches,
        'max_batch_size_mb': max_batch_size_mb
    }

# %%
# =============================================================================
# MAIN DRIVER FUNCTION FOR BATCH PROCESSING
# =============================================================================

async def batch_process_images_with_analysis(
    csv_path: str, 
    image_column: str = 'image_path',
    prompt: str = "Does this image show more than a foot of standing water? Yes or No.",
    model: str = "gpt-4o-mini",
    output_dir: str = None,
    sample_size: int = 100,
    auto_confirm: bool = False,
    sample_percentage: float = None,
    random_seed: int = 42
):
    """
    Complete driver function for batch processing images with pre-analysis
    
    Args:
        csv_path: Path to CSV file containing image paths
        image_column: Name of column containing image paths
        prompt: Prompt to send with each image
        model: Model to use (gpt-4o-mini, gpt-4o, etc.)
        output_dir: Output directory (auto-generated if None)
        sample_size: Number of images to sample for analysis
        auto_confirm: Skip user confirmation prompts
        sample_percentage: If provided, sample this % of dataset (None = use full dataset)
        random_seed: Fixed random seed for reproducible sampling
    """
    
    print("=" * 80)
    print("🌊 FLOOD DETECTION BATCH PROCESSING")
    print("=" * 80)
    
    try:
        # 0. Environment validation
        print("\n🔍 ENVIRONMENT VALIDATION")
        print("-" * 40)
        validate_environment()
        validate_model(model)
        validate_prompt(prompt)
        
        # 1. Validate input file and images
        print("\n📁 FILE VALIDATION")
        print("-" * 40)
        validation_results = validate_csv_and_images(csv_path, image_column, max_check=20)
        
        # Load and validate data
        df = pd.read_csv(csv_path)
        original_size = len(df)
        logger.info(f"Loaded dataset with {original_size:,} rows")
        
        if image_column not in df.columns:
            raise ValueError(f"Column '{image_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Rename column for consistency
        if image_column != 'image_path':
            df = df.rename(columns={image_column: 'image_path'})
        
        # Check for missing image paths
        missing_paths = df['image_path'].isna().sum()
        if missing_paths > 0:
            logger.warning(f"Found {missing_paths} rows with missing image paths")
            df = df.dropna(subset=['image_path'])
            logger.info(f"Proceeding with {len(df):,} valid image paths")
        
        if len(df) == 0:
            raise ValueError("No valid image paths remaining after cleanup")
        
        # Handle sampling
        if sample_percentage is not None:
            if sample_percentage <= 0 or sample_percentage > 100:
                raise ValueError(f"Sample percentage must be between 0 and 100, got {sample_percentage}")
            
            print(f"\n🎲 DATASET SAMPLING")
            print("-" * 40)
            df = sample_dataset(df, sample_percentage, random_seed)
            print(f"Sampled {len(df):,} images ({sample_percentage}%) from {original_size:,} total images")
        
        # 2. Pre-analysis
        print("\n📊 DATASET ANALYSIS")
        print("-" * 40)
        
        logger.info("Analyzing dataset to estimate batch requirements...")
        
        # Create temporary CSV for analysis if we sampled
        if sample_percentage is not None:
            temp_csv_path = csv_path.replace('.csv', f'_sample_{sample_percentage}pct.csv')
            df.to_csv(temp_csv_path, index=False)
            analysis = analyze_dataset_for_batching(temp_csv_path, sample_size)
            analysis_csv_path = temp_csv_path
        else:
            analysis = analyze_dataset_for_batching(csv_path, sample_size)
            analysis_csv_path = csv_path
        
        # Calculate costs and time estimates
        estimated_cost_per_image = 0.001  # Rough estimate for gpt-4o-mini vision
        if model in ["gpt-4o", "gpt-4-turbo"]:
            estimated_cost_per_image = 0.01
        elif model in ["o1-mini", "o1-preview"]:
            estimated_cost_per_image = 0.005
        
        total_estimated_cost = len(df) * estimated_cost_per_image
        batch_discount_cost = total_estimated_cost * 0.5  # 50% batch discount
        
        sample_info = ""
        if sample_percentage is not None:
            sample_info = f"   • Sample: {sample_percentage}% of original dataset ({original_size:,} total)\n"
        
        print(f"""
📈 ANALYSIS RESULTS:
{sample_info}   • Processing images: {len(df):,}
   • Average image size: {analysis['avg_image_size_mb']:.1f}MB
   • Estimated requests per batch: {analysis['estimated_requests_per_batch']:,}
   • Estimated total batches: {analysis['estimated_batches']}
   • Model: {model}
   
💰 COST ESTIMATES:
   • Real-time API cost: ~${total_estimated_cost:.2f}
   • Batch API cost (50% off): ~${batch_discount_cost:.2f}
   • Total savings: ~${total_estimated_cost - batch_discount_cost:.2f}
   
⏱️  TIME ESTIMATES:
   • Batch processing: Within 24 hours
   • File upload time: ~{analysis['estimated_batches'] * 0.1:.1f} minutes
        """)
        
        # 3. User confirmation
        if not auto_confirm:
            print("\n❓ CONFIRMATION")
            print("-" * 40)
            response = input(f"Proceed with batch processing {len(df):,} images? (y/N): ").lower().strip()
            if response not in ['y', 'yes']:
                print("❌ Operation cancelled by user")
                return None
        
        # 4. Setup output directory
        if output_dir is None:
            timestamp = int(time.time())
            sample_suffix = f"_sample{sample_percentage}pct" if sample_percentage else ""
            output_dir = f"batch_processing_{timestamp}{sample_suffix}"
        
        Path(output_dir).mkdir(exist_ok=True)
        logger.info(f"Using output directory: {output_dir}")
        
        # 5. Create and submit batches
        print("\n🚀 BATCH CREATION & SUBMISSION")
        print("-" * 40)
        
        batch_jobs = await process_large_dataset_batch(
            csv_path=analysis_csv_path,  # Use the (possibly sampled) CSV
            prompt=prompt,
            output_dir=output_dir,
            model=model
        )
        
        # Save job info
        job_info = {
            'csv_path': csv_path,
            'processed_csv_path': analysis_csv_path,
            'original_size': original_size,
            'processed_images': len(df),
            'sample_percentage': sample_percentage,
            'random_seed': random_seed,
            'model': model,
            'prompt': prompt,
            'batch_jobs': batch_jobs,
            'analysis': analysis,
            'validation_results': validation_results,
            'timestamp': timestamp
        }
        
        job_info_path = f"{output_dir}/job_info.json"
        with open(job_info_path, 'w') as f:
            json.dump(job_info, f, indent=2, default=str)
        
        print(f"""
✅ BATCH SUBMISSION COMPLETE!
   • Submitted {len(batch_jobs)} batches
   • Job info saved to: {job_info_path}
   • Batches will complete within 24 hours
   
📋 NEXT STEPS:
   1. Monitor progress: await monitor_all_batches(batch_jobs, '{output_dir}')
   2. Or check status later with batch IDs from job_info.json
        """)
        
        return batch_jobs
        
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
        print(f"❌ ERROR: {e}")
        print("Please fix the issues above before proceeding.")
        return None

async def quick_start_batch_processing():
    """Quick start example for batch processing"""
    
    # Example: Process your 929k images
    csv_path = input("Enter path to your CSV file: ").strip()
    
    if not csv_path or not os.path.exists(csv_path):
        print("❌ Invalid CSV path")
        return
    
    # Load CSV to check columns
    df_preview = pd.read_csv(csv_path, nrows=5)
    print(f"\nCSV Preview (first 5 rows):")
    print(df_preview.head())
    print(f"\nAvailable columns: {list(df_preview.columns)}")
    
    image_column = input(f"Enter image path column name (default: 'image_path'): ").strip()
    if not image_column:
        image_column = 'image_path'
    
    model = input(f"Enter model (default: 'gpt-4o-mini'): ").strip()
    if not model:
        model = 'gpt-4o-mini'
    
    prompt = input(f"Enter prompt (default: flood detection): ").strip()
    if not prompt:
        prompt = "Does this image show more than a foot of standing water? Yes or No."
    
    # Ask about sampling
    sample_input = input(f"Sample percentage of dataset? (default: no sampling, enter number 1-100): ").strip()
    sample_percentage = None
    if sample_input:
        try:
            sample_percentage = float(sample_input)
            if sample_percentage <= 0 or sample_percentage > 100:
                print("❌ Invalid sample percentage, using full dataset")
                sample_percentage = None
        except ValueError:
            print("❌ Invalid input, using full dataset")
            sample_percentage = None
    
    print(f"\n🚀 Starting batch processing...")
    if sample_percentage:
        print(f"📊 Will process {sample_percentage}% sample of the dataset")
    
    batch_jobs = await batch_process_images_with_analysis(
        csv_path=csv_path,
        image_column=image_column,
        prompt=prompt,
        model=model,
        sample_percentage=sample_percentage,
        auto_confirm=False
    )
    
    return batch_jobs

async def monitor_all_batches(batch_jobs: List[dict], output_dir: str = "batch_processing"):
    """Monitor all batches and download results when completed"""
    completed_results = []
    
    while len(completed_results) < len(batch_jobs):
        for job in batch_jobs:
            if job['batch_id'] in [r['batch_id'] for r in completed_results]:
                continue  # Already completed
            
            batch = await client.batches.retrieve(job['batch_id'])
            
            if batch.status == "completed":
                # Download results
                results_file = await download_batch_results(
                    job['batch_id'], 
                    f"{output_dir}/results/batch_{job['batch_id']}_results.jsonl"
                )
                
                completed_results.append({
                    'batch_id': job['batch_id'],
                    'results_file': results_file,
                    'start_idx': job['start_idx'],
                    'end_idx': job['end_idx']
                })
                
                logger.info(f"Completed batch {job['batch_id']} ({len(completed_results)}/{len(batch_jobs)})")
            
            elif batch.status in ["failed", "expired", "cancelled"]:
                logger.error(f"Batch {job['batch_id']} failed with status: {batch.status}")
                completed_results.append({
                    'batch_id': job['batch_id'],
                    'results_file': None,
                    'start_idx': job['start_idx'],
                    'end_idx': job['end_idx'],
                    'error': batch.status
                })
        
        if len(completed_results) < len(batch_jobs):
            logger.info(f"Waiting for {len(batch_jobs) - len(completed_results)} batches to complete...")
            await asyncio.sleep(300)  # Check every 5 minutes
    
    return completed_results

def combine_batch_results(completed_results: List[dict], original_df: pd.DataFrame, output_path: str):
    """Combine all batch results into final DataFrame"""
    all_responses = [''] * len(original_df)
    
    for result in completed_results:
        if result.get('results_file'):
            batch_df = parse_batch_results(result['results_file'])
            
            # Map results back to original indices
            for _, row in batch_df.iterrows():
                original_idx = result['start_idx'] + row['index']
                if original_idx < len(all_responses):
                    all_responses[original_idx] = row['response']
        else:
            # Handle failed batches
            start_idx = result['start_idx']
            end_idx = result['end_idx']
            error_msg = f"BATCH_ERROR: {result.get('error', 'unknown')}"
            for i in range(start_idx, end_idx):
                if i < len(all_responses):
                    all_responses[i] = error_msg
    
    # Add responses to original DataFrame
    final_df = original_df.copy()
    final_df['response'] = all_responses
    final_df.to_csv(output_path, index=False)
    
    logger.info(f"Combined results saved to {output_path}")
    return final_df

# %%
# =============================================================================
# UTILITY FUNCTIONS FOR MANAGING BATCH JOBS
# =============================================================================

async def check_batch_status(batch_id: str):
    """Check the status of a single batch"""
    try:
        batch = await client.batches.retrieve(batch_id)
        return {
            'batch_id': batch_id,
            'status': batch.status,
            'created_at': batch.created_at,
            'completed_at': getattr(batch, 'completed_at', None),
            'failed_at': getattr(batch, 'failed_at', None),
            'request_counts': getattr(batch, 'request_counts', {}),
            'errors': getattr(batch, 'errors', None)
        }
    except Exception as e:
        logger.error(f"Error checking batch {batch_id}: {e}")
        return {
            'batch_id': batch_id,
            'status': 'error',
            'error': str(e)
        }

async def check_all_batch_status(batch_jobs: List[dict]):
    """Check status of all batches and return summary"""
    print("\n📊 BATCH STATUS CHECK")
    print("-" * 50)
    
    status_counts = {}
    batch_statuses = []
    
    for job in batch_jobs:
        status_info = await check_batch_status(job['batch_id'])
        batch_statuses.append(status_info)
        
        status = status_info['status']
        status_counts[status] = status_counts.get(status, 0) + 1
        
        # Print status with appropriate emoji
        status_emoji = {
            'validating': '🔍',
            'in_progress': '⏳', 
            'finalizing': '🔄',
            'completed': '✅',
            'failed': '❌',
            'expired': '⏰',
            'cancelled': '🚫',
            'error': '💥'
        }
        
        emoji = status_emoji.get(status, '❓')
        print(f"{emoji} Batch {job['batch_id'][:8]}... - {status}")
        
        if status_info.get('request_counts'):
            counts = status_info['request_counts']
            print(f"   Total: {counts.get('total', 0)}, Completed: {counts.get('completed', 0)}, Failed: {counts.get('failed', 0)}")
    
    print(f"\n📈 SUMMARY:")
    for status, count in status_counts.items():
        emoji = status_emoji.get(status, '❓')
        print(f"{emoji} {status.title()}: {count}")
    
    return batch_statuses

def load_job_info(job_info_path: str):
    """Load job info from a previous run"""
    try:
        with open(job_info_path, 'r') as f:
            job_info = json.load(f)
        
        batch_jobs = job_info.get('batch_jobs', [])
        logger.info(f"Loaded job info with {len(batch_jobs)} batches")
        return job_info, batch_jobs
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Job info file not found: {job_info_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in job info file: {e}")

async def resume_monitoring(job_info_path: str):
    """Resume monitoring batches from a previous run"""
    job_info, batch_jobs = load_job_info(job_info_path)
    output_dir = os.path.dirname(job_info_path)
    
    print(f"📂 Resuming monitoring for {len(batch_jobs)} batches")
    print(f"📁 Output directory: {output_dir}")
    
    # Check current status
    await check_all_batch_status(batch_jobs)
    
    # Monitor until completion
    completed_results = await monitor_all_batches(batch_jobs, output_dir)
    
    # Combine results if all completed
    completed_count = sum(1 for r in completed_results if r.get('results_file'))
    if completed_count > 0:
        print(f"\n🔗 COMBINING RESULTS FROM {completed_count} COMPLETED BATCHES")
        
        # Load original data for combining
        original_df = pd.read_csv(job_info['processed_csv_path'])
        final_output_path = f"{output_dir}/final_results.csv"
        
        final_df = combine_batch_results(completed_results, original_df, final_output_path)
        
        # Print summary
        total_responses = (final_df['response'] != '').sum()
        error_responses = final_df['response'].str.startswith(('ERROR:', 'BATCH_ERROR:')).sum()
        
        print(f"✅ FINAL RESULTS SAVED: {final_output_path}")
        print(f"📊 Total responses: {total_responses:,}")
        print(f"❌ Errors: {error_responses:,}")
        print(f"✅ Success rate: {(total_responses-error_responses)/len(final_df)*100:.1f}%")
        
        return final_df
    
    return completed_results

# %%
# Example usage functions:

async def example_small_test():
    """Example: Test with a small sample"""
    batch_jobs = await batch_process_images_with_analysis(
        csv_path="your_dataset.csv",
        sample_percentage=1.0,  # Just 1% for testing
        auto_confirm=True
    )
    return batch_jobs

async def example_full_processing():
    """Example: Process full dataset"""
    batch_jobs = await batch_process_images_with_analysis(
        csv_path="your_929k_dataset.csv",
        prompt="Does this image show more than a foot of standing water? Yes or No.",
        model="gpt-4o-mini",
        auto_confirm=False  # Will ask for confirmation
    )
    return batch_jobs

async def example_resume_from_job_file():
    """Example: Resume monitoring from job info file"""
    results = await resume_monitoring("batch_processing_1234567890/job_info.json")
    return results

# %%
# FINAL USAGE INSTRUCTIONS:
"""
🚀 TO START BATCH PROCESSING:

1. Interactive mode:
   batch_jobs = await quick_start_batch_processing()

2. Direct usage:
   batch_jobs = await batch_process_images_with_analysis(
       csv_path="your_data.csv",
       image_column="image_path",
       model="gpt-4o-mini",
       sample_percentage=10.0  # Optional: process 10% sample
   )

3. Monitor progress:
   await check_all_batch_status(batch_jobs)
   
4. Monitor until completion:
   results = await monitor_all_batches(batch_jobs, "output_directory")
   
5. Resume from previous run:
   results = await resume_monitoring("batch_processing_xxx/job_info.json")

📁 All outputs are saved to timestamped directories with full job info for resuming.
"""

async def main():
    # Adjust these parameters for your dataset
    batch_jobs = await batch_process_images_with_analysis(
        csv_path='/share/ju/matt/street-flooding/notebooks/cambrian/entire_sep29_all.csv',
        image_column='image_path',
        prompt='Is the street in this image flooded? You are required to give a one-word, "Yes" or "No" response.',
        model='gpt-4o-mini',
        sample_percentage=0.1,  # Set to number (e.g., 10.0) for testing
        auto_confirm=False  # Changed to True for HPC
    )
    print(f'Submitted {len(batch_jobs)} batches successfully!')

if __name__ == "__main__":
    asyncio.run(main())
