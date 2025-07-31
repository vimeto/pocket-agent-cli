"""Model management service for downloading and managing GGUF models."""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import httpx
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from ..config import MODELS_DIR, DEFAULT_MODELS, Model


class ModelService:
    """Service for managing LLM models."""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.metadata_file = self.models_dir / "models.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load model metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                data = json.load(f)
                self.models = {m["id"]: Model(**m) for m in data}
        else:
            # Initialize with default models
            self.models = {m["id"]: Model(**m) for m in DEFAULT_MODELS}
            self._save_metadata()
    
    def _save_metadata(self) -> None:
        """Save model metadata to disk."""
        data = [m.model_dump() for m in self.models.values()]
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def list_models(self) -> List[Model]:
        """List all available models."""
        return list(self.models.values())
    
    def get_model(self, model_id: str) -> Optional[Model]:
        """Get a specific model by ID."""
        return self.models.get(model_id)
    
    def get_downloaded_models(self) -> List[Model]:
        """Get list of downloaded models."""
        return [m for m in self.models.values() if m.downloaded]
    
    async def download_model(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Model:
        """Download a model from Hugging Face.
        
        Args:
            model_id: ID of the model to download
            hf_token: Hugging Face token for gated models
            progress_callback: Callback for download progress (bytes_downloaded, total_bytes)
            
        Returns:
            Updated model with download status
        """
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        if model.downloaded and model.path and model.path.exists():
            return model
        
        # Create model filename
        filename = f"{model_id}.gguf"
        model_path = self.models_dir / filename
        
        if model.url and model.url.startswith("https://huggingface.co/"):
            # Parse HF URL to get repo and filename
            parts = model.url.replace("https://huggingface.co/", "").split("/")
            if len(parts) >= 5 and parts[2] == "resolve":
                repo_id = f"{parts[0]}/{parts[1]}"
                hf_filename = parts[-1]
                
                # Use huggingface_hub for authenticated downloads
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=hf_filename,
                        local_dir=str(self.models_dir),
                        local_dir_use_symlinks=False,
                        token=hf_token if model.requiresAuth else None,
                    )
                    
                    # Rename to our standard filename
                    final_path = Path(downloaded_path)
                    if final_path != model_path:
                        if model_path.exists():
                            model_path.unlink()
                        final_path.rename(model_path)
                    
                except Exception as e:
                    raise RuntimeError(f"Failed to download model: {e}")
            else:
                # Direct download with httpx
                await self._download_file(model.url, model_path, model.size, progress_callback)
        else:
            raise ValueError(f"Model {model_id} has no valid download URL")
        
        # Update model metadata
        model.downloaded = True
        model.path = model_path
        self.models[model_id] = model
        self._save_metadata()
        
        return model
    
    async def _download_file(
        self,
        url: str,
        path: Path,
        expected_size: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> None:
        """Download a file with progress tracking."""
        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                
                total_size = int(response.headers.get("content-length", 0))
                if expected_size and total_size != expected_size:
                    print(f"Warning: Expected size {expected_size}, got {total_size}")
                
                downloaded = 0
                with open(path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):  # 1MB chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback:
                            progress_callback(downloaded, total_size)
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model.
        
        Args:
            model_id: ID of the model to delete
            
        Returns:
            True if deleted, False if not found or not downloaded
        """
        model = self.models.get(model_id)
        if not model or not model.downloaded:
            return False
        
        if model.path and model.path.exists():
            model.path.unlink()
        
        model.downloaded = False
        model.path = None
        self.models[model_id] = model
        self._save_metadata()
        
        return True
    
    def add_custom_model(
        self,
        id: str,
        name: str,
        path: Path,
        architecture: str,
        quantization: Optional[str] = None,
    ) -> Model:
        """Add a custom model from a local file.
        
        Args:
            id: Unique identifier for the model
            name: Display name
            path: Path to the GGUF file
            architecture: Model architecture (llama, gemma, qwen, etc.)
            quantization: Quantization type (Q4_K_M, etc.)
            
        Returns:
            Created model
        """
        if not path.exists():
            raise ValueError(f"Model file not found: {path}")
        
        model = Model(
            id=id,
            name=name,
            size=path.stat().st_size,
            architecture=architecture,
            quantization=quantization,
            downloaded=True,
            path=path,
        )
        
        self.models[id] = model
        self._save_metadata()
        
        return model
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary with model information
        """
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        info = model.model_dump()
        
        # Add additional info if downloaded
        if model.downloaded and model.path and model.path.exists():
            info["actual_size"] = model.path.stat().st_size
            info["size_mb"] = round(info["actual_size"] / (1024 * 1024), 2)
        
        return info