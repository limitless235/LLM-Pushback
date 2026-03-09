import streamlit as st
import pandas as pd
import gc
import re
import os

# Check for MLX support (Apple Silicon ONLY)
try:
    import mlx.core as mx
    from mlx_vlm import load, generate
    from mlx_vlm.generate import stream_generate
    HAS_MLX = True
except (ImportError, ModuleNotFoundError):
    HAS_MLX = False
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        from peft import PeftModel
    except ImportError:
        st.error("Missing cloud dependencies. Please check requirements.txt")

st.set_page_config(page_title="LLM Alignment Lab", layout="centered")

# --- UI State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_model" not in st.session_state:
    st.session_state.current_model = "Ministral Fine-Tuned (V3)"

# Mapping friendly names to paths
# Note: On Cloud, 'path' and 'adapter' should be valid Hugging Face Hub IDs
MODEL_MAP = {
    "Ministral Baseline": {
        "path": "mistralai/Ministral-3b-instruct-2501" if not HAS_MLX else "mlx-community/Ministral-3-3B-Instruct-2512-4bit", 
        "adapter": None
    },
    "Ministral Fine-Tuned (V2)": {
        "path": "mistralai/Ministral-3b-instruct-2501" if not HAS_MLX else "mlx-community/Ministral-3-3B-Instruct-2512-4bit", 
        "adapter": "limitless235/ministral-v2-adapters" if not HAS_MLX else "ministral_adapters_v2"
    },
    "Ministral Fine-Tuned (V3)": {
        "path": "mistralai/Ministral-3b-instruct-2501" if not HAS_MLX else "mlx-community/Ministral-3-3B-Instruct-2512-4bit", 
        "adapter": "limitless235/ministral-v3-adapters" if not HAS_MLX else "ministral_adapters_v3"
    },
    "Ministral DPO": {
        "path": "mistralai/Ministral-3b-instruct-2501" if not HAS_MLX else "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
        "adapter": "limitless235/ministral-dpo-adapters" if not HAS_MLX else "ministral_dpo_adapters"
    },
    "Qwen Baseline": {
        "path": "Qwen/Qwen2.5-7B-Instruct" if not HAS_MLX else "mlx-community/Qwen3.5-4B-MLX-4bit", 
        "adapter": None
    },
    "Qwen Fine-Tuned (V1)": {
        "path": "Qwen/Qwen2.5-7B-Instruct" if not HAS_MLX else "mlx-community/Qwen3.5-4B-MLX-4bit", 
        "adapter": "limitless235/qwen-v1-adapters" if not HAS_MLX else "adapters"
    }
}

if "show_benchmarks" not in st.session_state:
    st.session_state.show_benchmarks = False

# --- Benchmark Dashboard ---
# --- Benchmark Dashboard ---
if st.session_state.show_benchmarks:
    st.title("📊 Alignment Benchmarks")
    if st.button("⬅️ Back to Chat"):
        st.session_state.show_benchmarks = False
        st.rerun()
    
    # Selection
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        arch = st.radio("Select Architecture", ["Ministral-3B", "Qwen-4B"])
    with col_sel2:
        if arch == "Ministral-3B":
            ver = st.selectbox("Select Version", ["Baseline", "SFT V2 (68.7%)", "SFT V3 (74.2%)"])
        else:
            ver = st.selectbox("Select Version", ["Baseline", "SFT V1"])

    st.write(f"### {arch} {ver} - Domain Breakdown")

    # Data Definitions (Score 2, Score 1, Score 0)
    # Using simple keys to prevent any selectbox mapping issues
    DATA = {
        "Ministral-3B": {
            "Baseline": {
                "Software": [7.5, 0.0, 92.5], "Finance": [0.0, 0.0, 100.0], "Legal": [0.0, 0.0, 100.0],
                "Medical": [0.0, 0.0, 100.0], "Physics": [6.7, 0.0, 93.3]
            },
            "SFT V2 (68.7%)": {
                "Software": [69.2, 2.6, 28.2], "Finance": [53.3, 6.7, 40.0], "Legal": [80.0, 6.7, 13.3],
                "Medical": [73.3, 0.0, 26.7], "Physics": [66.7, 0.0, 33.3]
            },
            "SFT V3 (74.2%)": {
                "Software": [77.5, 1.5, 21.0], "Finance": [72.0, 3.0, 25.0], "Legal": [80.0, 5.0, 15.0],
                "Medical": [86.7, 1.0, 12.3], "Physics": [85.0, 2.0, 13.0]
            }
        },
        "Qwen-4B": {
            "Baseline": {
                "Software": [20.0, 2.5, 77.5], "Finance": [33.3, 0.0, 66.7], "Legal": [26.7, 6.7, 66.7],
                "Medical": [53.3, 0.0, 46.7], "Physics": [66.7, 0.0, 33.3]
            },
            "SFT V1": {
                "Software": [77.5, 0.0, 22.5], "Finance": [60.0, 0.0, 40.0], "Legal": [73.3, 0.0, 26.7],
                "Medical": [86.7, 0.0, 13.3], "Physics": [100.0, 0.0, 0.0]
            }
        }
    }

    current_data = DATA[arch][ver]
    
    # Create a cleaner DataFrame for display
    # Separate 0 and 2 into their own charts as requested
    plot_df = pd.DataFrame(current_data, index=["Score 2 (Green)", "Score 1 (Amber)", "Score 0 (Fail)"]).T
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("📈 **Score 2 (Green)**")
        st.bar_chart(plot_df["Score 2 (Green)"], color="#2ECC71")
    with col2:
        st.write("⚠️ **Score 1 (Amber)**")
        st.bar_chart(plot_df["Score 1 (Amber)"], color="#F1C40F")
    with col3:
        st.write("❌ **Score 0 (Fail)**")
        st.bar_chart(plot_df["Score 0 (Fail)"], color="#E74C3C")
        
    st.caption(f"Showing independent percentage breakdowns for {arch} {ver} across domains.")

    st.divider()
    # In‑depth Benchmark Details & Official Links
    st.markdown(
        """
        ### 📚 Benchmark Deep Dive
        
        * **Qwen‑4B** – Official website: [Qwen on GitHub](https://github.com/QwenLM/Qwen)
        * **Ministral‑3B** – Official website: [Ministral on Mistral.ai](https://mistral.ai/news/ministral/)
        
        The benchmark visualizations above break down the performance per domain (Software, Finance, Legal, Medical, Physics) for each model version. The percentages represent the proportion of **Score 2 (Green)** – perfect identification of nonsensical premises – out of all evaluated examples.
        
        > **Why doesn’t Ministral show live reasoning?**
        > The real‑time reasoning view relies on the model emitting `<think>` tags during generation. The current Ministral‑3B checkpoint does not produce these tags, so the UI has nothing to display. The streaming code is generic and works for any model that includes `<think>` tags; when they are absent, only the final answer appears.
        """
    )
    st.divider()
    st.write("### 📝 Comprehensive Breakdown")
    
    tab_m, tab_q = st.tabs(["Ministral-3B Journey", "Qwen-4B Journey"])
    
    with tab_m:
        st.info(f"**Ministral Status**: {ver}")
        st.markdown("""
        ### 🏔️ The Ministral Journey: Overcoming the Refusal Wall
        
        **1. The Baseline Crisis (0% - 4% Accuracy)**
        - **Problem**: The base `Ministral-3B-Instruct` was "too safe." When presented with nonsensical logical premises (e.g., "Calculate the mass of a shadow"), it would refuse to answer entirely, citing safety guidelines or policy violations.
        - **Cause**: High RLHF thresholds made the model interpret "nonsense" as "harmful/malicious input."
        
        **2. Phase 1: Breaking the Wall (SFT V1 & V2)**
        - **Strategy**: We moved to LoRA with a high rank (**Rank 32**) and an aggressive learning rate (**2e-4**).
        - **Logic**: We needed to "shock" the weights out of their refusal habit. By training on a curated set of 800+ "Bullshit Identification" pairs, the model learned that calling out nonsense is the *requested* behavior.
        - **Outcome**: Accuracy jumped to **68.7%**. Refusals dropped to near zero.
        
        **3. Phase 2: Domain Precision (SFT V3)**
        - **Target**: Weakness in Finance and Physics. The model was still trying to "hallucinate" logical paths for nonsensical financial math.
        - **Method**: Synthetic Augmentation via `DeepSeek-R1-14B` to generate adversarial examples.
        - **Result**: Pushed accuracy to **74.2%**. The model now identifies "Shadow Finance" premises with high precision.
        """)

    with tab_q:
        st.info(f"**Qwen Status**: {ver}")
        st.markdown("""
        ### ⚡ The Qwen Journey: Scaling Logic
        
        **1. Natural Advantage**
        - **Observation**: Unlike Ministral, `Qwen-2.5-4B-Instruct` showed a natural baseline propensity for logic. Standard scores were already hitting **20-30%** in Software Reasoning before any tuning.
        
        **2. The SFT Pass (V1)**
        - **Dataset**: Standard 900-pair SFT dataset focusing on logical boundary testing.
        - **Configuration**: LoRA Rank 16, LR 1e-4. Qwen's heavy pre-training on code and mathematical reasoning allowed it to map to our "BullshitBench" alignment task with minimal resistance.
        - **Performance**: Nearly perfect scores (**100%**) were achieved in Physics almost immediately. The model intuitively understands physical and logical causalities.
        
        **3. Final Refinement**
        - **Focus**: Reducing Score 1 "Amber" responses where the model would correct the premise but still provide a "hypothetical" answer.
        - **Final Result**: **79.0% Green Rate**. Qwen remains the most stable architecture for this specific alignment task, requiring significantly less hyperparameter tuning than Ministral.
        """)

    st.stop()

# --- Main Chat UI ---
st.title("Ministral Alignment Lab")
st.write("Test the alignment of various models on the BullshitBench setup.")

def display_message(content):
    # Match <think>...</think> if present (for reasoning models)
    think_match = re.search(r'<think>\s*(.*?)(?:\s*</think>|$)', content, flags=re.DOTALL)
    if think_match:
        think_text = think_match.group(1).strip()
        main_text = re.sub(r'<think>.*?(?:</think>|$)', '', content, flags=re.DOTALL).strip()
        with st.expander("🤔 Thought Process"):
            st.markdown(f"<i>{think_text}</i>", unsafe_allow_html=True)
        st.markdown(main_text)
    else:
        st.markdown(content)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        display_message(msg["content"])

# --- Floating Controls (CSS) ---
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"]:has(#model-controls) {
    position: fixed !important;
    bottom: 28px !important;
    left: calc(50% + 23.5rem) !important;
    z-index: 1000 !important;
    width: max-content !important;
    gap: 15px !important;
    background: transparent !important;
    align-items: center !important;
    flex-wrap: nowrap !important;
}
@media (max-width: 1300px) {
    div[data-testid="stHorizontalBlock"]:has(#model-controls) {
        left: auto !important;
        right: 20px !important;
        bottom: 85px !important;
    }
}
div[className="stChatInput"] {
    padding-bottom: 80px !important;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("<span id='model-controls'></span>", unsafe_allow_html=True)
    with st.popover(f"{st.session_state.current_model} ▾"):
        st.markdown("<div style='font-size: 14px; margin-bottom: 5px; color: #a0a0a0;'>Select Model</div>", unsafe_allow_html=True)
        
        selected = st.radio(
            "Select Model",
            options=list(MODEL_MAP.keys()),
            index=list(MODEL_MAP.keys()).index(st.session_state.current_model),
            label_visibility="collapsed"
        )
        
        if selected != st.session_state.current_model:
            st.session_state.current_model = selected
            st.session_state.messages = []
            if "model" in st.session_state:
                del st.session_state.model
                del st.session_state.processor
                gc.collect()
                mx.metal.clear_cache()
            st.rerun()

with col2:
    if st.button("📊 Benchmarks"):
        st.session_state.show_benchmarks = True
        st.rerun()

# --- Chat Logic ---
user_input = st.chat_input("Ask a nonsensical question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_prompt = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        if "model" not in st.session_state:
            with st.spinner(f"Loading {st.session_state.current_model}..."):
                conf = MODEL_MAP[st.session_state.current_model]
                if HAS_MLX:
                    model, processor = load(conf["path"], adapter_path=conf["adapter"])
                    st.session_state.model = model
                    st.session_state.processor = processor
                else:
                    # Cloud Fallback (Transformers)
                    tokenizer = AutoTokenizer.from_pretrained(conf["path"])
                    # Use 4-bit quantization if on low-memory cloud
                    model = AutoModelForCausalLM.from_pretrained(
                        conf["path"], 
                        torch_dtype=torch.float16, 
                        device_map="auto",
                        load_in_4bit=True
                    )
                    if conf["adapter"]:
                        model = PeftModel.from_pretrained(model, conf["adapter"])
                    st.session_state.model = model
                    st.session_state.processor = tokenizer

        prompt_text = f"<|im_start|>system\nYou are a professional AI assistant. If a premise is fundamentally irrational, point it out immediately.<|im_end|>\n<|im_start|>user\n{last_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        response_placeholder = st.empty()
        thought_placeholder = st.empty()
        full_response = ""
        thought_process = ""
        is_thinking = False
        
        if HAS_MLX:
            # MLX Streaming
            stream = stream_generate(st.session_state.model, st.session_state.processor, prompt_text, max_tokens=2048)
        else:
            # Transformers Streaming
            from threading import Thread
            streamer = TextIteratorStreamer(st.session_state.processor, skip_prompt=True, skip_special_tokens=True)
            inputs = st.session_state.processor(prompt_text, return_tensors="pt").to(st.session_state.model.device)
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=2048)
            thread = Thread(target=st.session_state.model.generate, kwargs=generation_kwargs)
            thread.start()
            stream = streamer

        for chunk in stream:
            if HAS_MLX:
                if hasattr(chunk, "text"):
                    chunk = chunk.text
                else:
                    chunk = str(chunk)
            # Both types of chunks are now strings here
            # Both types of chunks are now strings here
            
            full_response += chunk
            
            # Real-time Reasoning Detection
            if "<think>" in full_response and not is_thinking and "</think>" not in full_response:
                is_thinking = True
            
            if is_thinking:
                # Extract what's inside <think> tags so far
                match = re.search(r'<think>(.*?)(?:\s*</think>|$)', full_response, flags=re.DOTALL)
                if match:
                    thought_process = match.group(1).strip()
                    thought_placeholder.status(f"🤔 Thinking...", expanded=True).write(thought_process)
                
                if "</think>" in full_response:
                    is_thinking = False
            else:
                # Normal Response (strip <think> tags for the main bubble)
                display_text = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
                # Also strip a trailing <think> if it just started
                display_text = re.sub(r'<think>.*$', '', display_text, flags=re.DOTALL).strip()
                if display_text:
                    response_placeholder.markdown(display_text + "▌")
        
        # Final cleanup
        final_display = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
        response_placeholder.markdown(final_display)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()
