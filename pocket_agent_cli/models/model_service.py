"""Model management service for downloading and managing GGUF models with versioning support."""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import httpx
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from ..config import MODELS_DIR, DEFAULT_MODELS, Model, ModelVersion


class ModelService:
    """Service for managing LLM models with versioning support."""

    def __init__(self):
        self.models_dir = MODELS_DIR
        self.metadata_file = self.models_dir / "models.json"
        self._load_metadata()
        self._migrate_legacy_models()

    def _migrate_legacy_models(self) -> None:
        """Migrate legacy models to the new versioning system."""
        updated = False
        for model_id, model in self.models.items():
            # Check if model uses old format (has url but no versions)
            if model.url and not model.versions:
                # Create a version from legacy fields
                version = ModelVersion(
                    url=model.url,
                    size=model.size or 0,
                    quantization=model.quantization or "Q4_K_M",
                    downloaded=model.downloaded,
                    path=model.path
                )
                model.versions = {"Q4_K_M": version}
                model.default_version = "Q4_K_M"
                updated = True
                
        if updated:
            self._save_metadata()

    def refresh_from_defaults(self) -> None:
        """Refresh model metadata from DEFAULT_MODELS while preserving download status."""
        updated = False
        
        # Process each model in DEFAULT_MODELS
        for default_model in DEFAULT_MODELS:
            model_id = default_model["id"]
            
            if model_id in self.models:
                # Existing model - update with new versions while preserving download status
                existing_model = self.models[model_id]
                
                # Update basic metadata (name, architecture, etc)
                existing_model.name = default_model["name"]
                existing_model.architecture = default_model["architecture"]
                existing_model.requiresAuth = default_model.get("requiresAuth", False)
                
                # Check for new versions
                for v_name, v_data in default_model.get("versions", {}).items():
                    if v_name not in existing_model.versions:
                        # Add new version
                        existing_model.versions[v_name] = ModelVersion(**v_data)
                        updated = True
                    else:
                        # Update existing version URL/size if changed, preserve download status
                        existing_version = existing_model.versions[v_name]
                        if existing_version.url != v_data["url"] or existing_version.size != v_data["size"]:
                            existing_version.url = v_data["url"]
                            existing_version.size = v_data["size"]
                            existing_version.quantization = v_data["quantization"]
                            # Keep downloaded status and path unchanged
                            updated = True
                
                # Remove versions that are no longer in DEFAULT_MODELS
                versions_to_remove = []
                for v_name in existing_model.versions:
                    if v_name not in default_model.get("versions", {}):
                        versions_to_remove.append(v_name)
                
                for v_name in versions_to_remove:
                    del existing_model.versions[v_name]
                    if versions_to_remove:
                        updated = True
                
            else:
                # New model - add it
                versions = {}
                for v_name, v_data in default_model.get("versions", {}).items():
                    versions[v_name] = ModelVersion(**v_data)
                
                m_copy = default_model.copy()
                m_copy["versions"] = versions
                self.models[model_id] = Model(**m_copy)
                updated = True
        
        # Remove models that are no longer in DEFAULT_MODELS (optional - you might want to keep them)
        # Commenting this out to preserve custom/removed models
        # models_to_remove = []
        # default_ids = {m["id"] for m in DEFAULT_MODELS}
        # for model_id in self.models:
        #     if model_id not in default_ids:
        #         models_to_remove.append(model_id)
        # for model_id in models_to_remove:
        #     del self.models[model_id]
        #     updated = True
        
        if updated:
            self._save_metadata()

    def _load_metadata(self) -> None:
        """Load model metadata from disk."""
        print(f"loading metadata from file #{self.metadata_file.exists()} #{self.metadata_file}")
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                data = json.load(f)
                self.models = {}
                for m in data:
                    # Handle both old and new format
                    if isinstance(m.get("versions"), dict):
                        # New format with versions
                        versions = {}
                        for v_name, v_data in m["versions"].items():
                            if isinstance(v_data, dict):
                                # Convert path strings back to Path objects
                                if v_data.get("path"):
                                    v_data["path"] = Path(v_data["path"])
                                versions[v_name] = ModelVersion(**v_data)
                        m["versions"] = versions
                    # Convert legacy path to Path object
                    if m.get("path"):
                        m["path"] = Path(m["path"])
                    self.models[m["id"]] = Model(**m)
            
            # Update models with new versions from DEFAULT_MODELS
            for default_model in DEFAULT_MODELS:
                model_id = default_model["id"]
                if model_id in self.models:
                    # Check for new versions in DEFAULT_MODELS
                    for v_name, v_data in default_model.get("versions", {}).items():
                        if v_name not in self.models[model_id].versions:
                            # Add new version
                            self.models[model_id].versions[v_name] = ModelVersion(**v_data)
                else:
                    # New model, add it
                    versions = {}
                    for v_name, v_data in default_model.get("versions", {}).items():
                        versions[v_name] = ModelVersion(**v_data)
                    m_copy = default_model.copy()
                    m_copy["versions"] = versions
                    self.models[model_id] = Model(**m_copy)
            
            # Save updated metadata
            self._save_metadata()
        else:
            # Initialize with default models
            self.models = {}
            for m in DEFAULT_MODELS:
                # Convert version dicts to ModelVersion objects
                versions = {}
                for v_name, v_data in m.get("versions", {}).items():
                    versions[v_name] = ModelVersion(**v_data)
                m_copy = m.copy()
                m_copy["versions"] = versions
                self.models[m["id"]] = Model(**m_copy)
            self._save_metadata()

    def _save_metadata(self) -> None:
        """Save model metadata to disk."""
        data = []
        for model in self.models.values():
            m_dict = model.model_dump()
            # Convert ModelVersion objects to dicts for JSON serialization
            if m_dict.get("versions"):
                versions_dict = {}
                for v_name, v_obj in m_dict["versions"].items():
                    if isinstance(v_obj, ModelVersion):
                        v_dict = v_obj.model_dump()
                    else:
                        v_dict = v_obj
                    # Convert Path to string for JSON
                    if v_dict.get("path"):
                        v_dict["path"] = str(v_dict["path"])
                    versions_dict[v_name] = v_dict
                m_dict["versions"] = versions_dict
            # Convert Path to string for legacy path field
            if m_dict.get("path"):
                m_dict["path"] = str(m_dict["path"])
            data.append(m_dict)
        
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def list_models(self) -> List[Model]:
        """List all available models."""
        return list(self.models.values())

    def get_model(self, model_id: str, version: Optional[str] = None) -> Optional[Model]:
        """Get a specific model by ID, optionally with a specific version set as current."""
        model = self.models.get(model_id)
        if model and version:
            # Validate version exists
            if version in model.versions:
                model.current_version = version
                # Update legacy fields for backward compatibility
                v = model.versions[version]
                model.url = v.url
                model.size = v.size
                model.quantization = v.quantization
                model.downloaded = v.downloaded
                model.path = v.path
        return model

    def get_downloaded_models(self) -> List[Dict[str, Any]]:
        """Get list of downloaded models with their versions."""
        downloaded = []
        for model in self.models.values():
            for version_name, version in model.versions.items():
                if version.downloaded:
                    downloaded.append({
                        "model": model,
                        "version": version_name,
                        "quantization": version.quantization,
                        "path": version.path
                    })
        return downloaded

    def list_model_versions(self, model_id: str) -> Dict[str, ModelVersion]:
        """List all available versions for a model."""
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        return model.versions

    async def download_model(
        self,
        model_id: str,
        version: Optional[str] = None,
        hf_token: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Model:
        """Download a specific version of a model from Hugging Face.

        Args:
            model_id: ID of the model to download
            version: Version to download (Q4_K_M, F16, etc). If None, uses default.
            hf_token: Hugging Face token for gated models
            progress_callback: Callback for download progress (bytes_downloaded, total_bytes)

        Returns:
            Updated model with download status
        """
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        # Get the version to download
        version = version or model.default_version
        if version not in model.versions:
            raise ValueError(f"Version {version} not found for model {model_id}")
        
        model_version = model.versions[version]
        
        # Check if already downloaded
        if model_version.downloaded and model_version.path and model_version.path.exists():
            model.current_version = version
            return model

        # Create model filename with version
        filename = f"{model_id}_{version}.gguf"
        model_path = self.models_dir / filename

        if model_version.url and model_version.url.startswith("https://huggingface.co/"):
            # Parse HF URL to get repo and filename
            parts = model_version.url.replace("https://huggingface.co/", "").split("/")
            if len(parts) >= 5 and parts[2] == "resolve":
                repo_id = f"{parts[0]}/{parts[1]}"
                hf_filename = parts[-1]

                # Use huggingface_hub for authenticated downloads
                # First check if file is cached
                try:
                    from huggingface_hub import try_to_load_from_cache
                    
                    cached_file = try_to_load_from_cache(
                        repo_id=repo_id,
                        filename=hf_filename,
                        cache_dir=None,
                    )
                    
                    if cached_file and isinstance(cached_file, str) and Path(cached_file).exists():
                        # File is already cached, just copy/link it
                        print(f"Using cached file: {cached_file}")
                        if progress_callback:
                            progress_callback(100, 100)  # Show complete
                        
                        # Copy or link to our location
                        import shutil
                        shutil.copy2(cached_file, model_path)
                    else:
                        # Need to download - use httpx with progress
                        # First get the actual download URL
                        from huggingface_hub import hf_hub_url
                        
                        download_url = hf_hub_url(
                            repo_id=repo_id,
                            filename=hf_filename,
                            repo_type="model"
                        )
                        
                        # Add auth header if needed
                        headers = {}
                        if model.requiresAuth and hf_token:
                            headers["Authorization"] = f"Bearer {hf_token}"
                        
                        # Download with progress using httpx
                        # We're already in an async context, so just await
                        await self._download_file_with_auth(
                            download_url, 
                            model_path, 
                            model_version.size,
                            progress_callback,
                            headers
                        )

                except Exception as e:
                    raise RuntimeError(f"Failed to download model: {e}")
            else:
                # Direct download with httpx
                await self._download_file(model_version.url, model_path, model_version.size, progress_callback)
        else:
            raise ValueError(f"Model {model_id} version {version} has no valid download URL")

        # Update model version metadata
        model_version.downloaded = True
        model_version.path = model_path
        model.versions[version] = model_version
        model.current_version = version
        
        # Update legacy fields for backward compatibility
        model.downloaded = True
        model.path = model_path
        model.url = model_version.url
        model.size = model_version.size
        model.quantization = model_version.quantization
        
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
    
    async def _download_file_with_auth(
        self,
        url: str,
        path: Path,
        expected_size: Optional[int] = None,
        progress_callback: Optional[callable] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Download a file with progress tracking and optional auth headers."""
        async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
            async with client.stream("GET", url, headers=headers or {}) as response:
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

    def delete_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """Delete a downloaded model version.

        Args:
            model_id: ID of the model
            version: Specific version to delete. If None, deletes all versions.

        Returns:
            True if deleted, False if not found or not downloaded
        """
        model = self.models.get(model_id)
        if not model:
            return False

        if version:
            # Delete specific version
            if version not in model.versions:
                return False
            
            model_version = model.versions[version]
            if not model_version.downloaded:
                return False
            
            if model_version.path and model_version.path.exists():
                model_version.path.unlink()
            
            model_version.downloaded = False
            model_version.path = None
            
            # Update legacy fields if this was the current version
            if model.current_version == version:
                model.current_version = None
                model.downloaded = False
                model.path = None
        else:
            # Delete all versions
            for v_name, v in model.versions.items():
                if v.downloaded and v.path and v.path.exists():
                    v.path.unlink()
                v.downloaded = False
                v.path = None
            
            model.downloaded = False
            model.path = None
            model.current_version = None

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
        version_name: Optional[str] = None,
    ) -> Model:
        """Add a custom model from a local file.

        Args:
            id: Unique identifier for the model
            name: Display name
            path: Path to the GGUF file
            architecture: Model architecture (llama, gemma, qwen, etc.)
            quantization: Quantization type (Q4_K_M, F16, etc.)
            version_name: Name for this version (defaults to quantization or "custom")

        Returns:
            Created or updated model
        """
        if not path.exists():
            raise ValueError(f"Model file not found: {path}")

        version_name = version_name or quantization or "custom"
        
        # Check if model already exists
        if id in self.models:
            model = self.models[id]
        else:
            model = Model(
                id=id,
                name=name,
                architecture=architecture,
                versions={},
                default_version=version_name
            )

        # Add the version
        version = ModelVersion(
            url=f"file://{path}",
            size=path.stat().st_size,
            quantization=quantization or "unknown",
            downloaded=True,
            path=path
        )
        
        model.versions[version_name] = version
        model.current_version = version_name
        
        # Update legacy fields
        model.downloaded = True
        model.path = path
        model.size = version.size
        model.quantization = version.quantization
        model.url = version.url

        self.models[id] = model
        self._save_metadata()

        return model

    def get_model_info(self, model_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about a model and its versions.

        Args:
            model_id: ID of the model
            version: Specific version to get info for. If None, shows all versions.

        Returns:
            Dictionary with model information
        """
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        info = {
            "id": model.id,
            "name": model.name,
            "architecture": model.architecture,
            "requiresAuth": model.requiresAuth,
            "default_version": model.default_version,
            "current_version": model.current_version,
            "versions": {}
        }

        if version:
            # Info for specific version
            if version not in model.versions:
                raise ValueError(f"Version {version} not found for model {model_id}")
            
            v = model.versions[version]
            info["versions"][version] = {
                "quantization": v.quantization,
                "size": v.size,
                "size_mb": round(v.size / (1024 * 1024), 2),
                "downloaded": v.downloaded,
                "path": str(v.path) if v.path else None
            }
            
            if v.downloaded and v.path and v.path.exists():
                info["versions"][version]["actual_size"] = v.path.stat().st_size
                info["versions"][version]["actual_size_mb"] = round(v.path.stat().st_size / (1024 * 1024), 2)
        else:
            # Info for all versions
            for v_name, v in model.versions.items():
                v_info = {
                    "quantization": v.quantization,
                    "size": v.size,
                    "size_mb": round(v.size / (1024 * 1024), 2),
                    "downloaded": v.downloaded,
                    "path": str(v.path) if v.path else None
                }
                
                if v.downloaded and v.path and v.path.exists():
                    v_info["actual_size"] = v.path.stat().st_size
                    v_info["actual_size_mb"] = round(v.path.stat().st_size / (1024 * 1024), 2)
                
                info["versions"][v_name] = v_info

        return info