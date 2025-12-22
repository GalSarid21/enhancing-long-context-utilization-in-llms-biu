from typing import List
from vllm import LLM, SamplingParams


class vLLM:
    """
    A wrapper class to abstract the vLLM package from the project.
    """

    def __init__(
        self,
        model: str,
        dtype: str,
        top_p: float,
        num_gpus: int,
        max_tokens: int,
        temperature: float,
        max_model_len: int,
        gpu_memory_utilization: float,
        
    ) -> None:
        
        self._model = model

        self._sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )

        self._llm = LLM(
            model=model,
            dtype=dtype,
            tensor_parallel_size=num_gpus,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization
        )

    async def generate_batch(self, prompts: List[str]) -> List[str]:
        results = self._llm.generate(prompts, self._sampling_params)
        return [res.outputs[0].text for res in results]
    
    async def generate(self, prompt: str,) -> str:
        return await self.generate_batch(prompts=[prompt])[0]
    
    async def shutdown_gracefully(self) -> None:
        try:
            self._llm.shutdown()
        except AttributeError:
            # fallback: try engine directly if exposed
            engine = getattr(self._llm, "llm_engine", None)
            if engine is not None and hasattr(engine, "shutdown"):
                engine.shutdown()
