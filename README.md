# ComfyUI-Fossiel-QoL-Nodes

ComfyUI-Fossiel-QoL-Nodes is (what will hopefully become) a suite of custom nodes for ComfyUI to improve quality of life. These nodes have been developed for personal use but maybe you’ll find them useful as well.

## Nodes

### FossielCentralControl_v2

![FossielCentralControl_v2](images/fcc_v2_ss.png)

This node is your go-to hub for managing settings in WAN 2.2 (MoE) video generation, especially on budget laptops. It simplifies creating video clips in multiple runs (like extending a sequence) and works great for any ComfyUI project, keeping things organized and easy.

#### Specifications

**Inputs:**
1. **Images (optional)** – Accepts a single image or a batch of images. It grabs the batch size (how many images) and the resolution (width and height) of the first image. The batch size goes to the `batch_count` output, and the resolution is used to set the target dimensions for video or image generation, overriding the Resolution Target setting if connected.

**Parameters:**
1. **Resolution Settings** – Controls the size of your output image or video.  
   - *Manual Resolution*: Set your preferred width and height (e.g., 512x512). These go straight to the `Manual Resolution` output for use in other nodes.  
   - *Resolution Target*: Define the ideal width and height for your project (e.g., 512x512). If an image is connected to the Images input, its dimensions take over. The node calculates a final resolution that matches this aspect ratio as closely as possible, stays within a maximum pixel limit (set by Maximum Resolution), and aligns with a tolerance value for better AI compatibility. Lower tolerance values give a more accurate aspect ratio, while higher ones work better with the AI’s internal math.  
   - *Maximum Resolution*: Caps the width and height to keep the total pixels manageable for low-spec hardware. The calculated resolution is sent to the `Calculated Resolution` output.  
   - *Note*: If calculations go wonky (e.g., zero height), it defaults to 64x64 to keep things safe.  
2. **Video Settings** – Manages video-specific options for smooth generation.  
   - *Video Length*: Sets the number of frames (e.g., 81). It snaps to 1 or multiples of 4 + 1 (like 5, 9, 13) for WAN 2.2 compatibility and goes to the `Vid_Length` output.  
   - *Overlap Length*: Sets how many frames overlap between video segments (e.g., 8), sent to the `Overlap` output.  
   - *Frame Rate*: Defines frames per second (e.g., 16.0), sent to the `Frame_Rate` output.  
   - *Start/End Frames*: Based on Video Length and Current Generation (e.g., which clip you’re making, like clip 1 or 2), it calculates the first and last frame numbers, sent to `Start_Index` and `End_Index`. It also computes their timings (in seconds) for `Start_Time` and `End_Time`.  
   - *First Generation Switch*: Outputs a yes/no signal (1 for the first clip, 0 for others) to tweak setups for the first video clip differently, sent to `First_Gen_Batch_Switch`.  
3. **Sampling and Naming Settings** – Handles AI generation and file naming.  
   - *Sampling Steps*: Sets the number of AI processing steps (snaps to even numbers, e.g., 4). It splits these steps for two samplers in WAN 2.2 workflows, using a split method (repeat last step or move to next) and a ratio (e.g., 50/50). These go to `steps`, `KSampler_1_End_Step`, and `KSampler_2_Start_Step`.  
   - *CFG Scale*: Controls how closely the AI follows your prompt (e.g., 7.0), sent to the `cfg` output.  
   - *Sampler and Scheduler*: Chooses the AI’s processing style (e.g., “euler” sampler, “simple” scheduler), sent to `Sampler` and `Scheduler`.  
   - *File Naming*: Lets you set a base name (e.g., “MyVideo”). You can add a generation number suffix (e.g., “_Gen001”) based on Current Generation (e.g., clip 1, 2, etc.), using a separator (e.g., “_”). The final name goes to `File_Name`, and the generation number to `Gen_Num`.  

**Outputs:**
1. **Manual Resolution** – The width and height you set in Resolution Settings, passed directly for use in other nodes (e.g., setting canvas size).  
2. **Calculated Resolution** – The final width and height calculated to match your target aspect ratio, stay within the pixel limit, and align with AI compatibility rules. Used for actual image or video generation.  
3. **Vid_Length** – The snapped number of video frames (e.g., 81).  
4. **Vid_Duration** – The video length in seconds (e.g., 81 frames at 16 FPS = 5.0625 seconds).  
5. **Overlap** – The number of overlapping frames between video segments.  
6. **Overlap_Duration** – The overlap time in seconds.  
7. **Start_Index** – The first frame number of the video sequence.  
8. **Start_Time** – The start time of the sequence in seconds.  
9. **End_Index** – The last frame number of the sequence.  
10. **End_Time** – The end time of the sequence in seconds.  
11. **Frame_Rate** – The frames per second for the video.  
12. **Gen_Num** – The current generation number (e.g., clip 1, 2).  
13. **batch_count** – The number of images in the input batch (0 if no image).  
14. **File_Name** – The final file name, with or without a generation suffix.  
15. **steps** – The total number of sampling steps (even).  
16. **KSampler_1_End_Step** – The steps for the first sampler in a dual-sampler setup.  
17. **KSampler_2_Start_Step** – The starting step for the second sampler.  
18. **cfg** – The guidance scale for the AI.  
19. **Sampler** – The chosen sampling algorithm.  
20. **Scheduler** – The chosen scheduler algorithm.  
21. **First_Gen_Batch_Switch** – A yes/no signal (1 or 0) for first-clip setup adjustments.

**Notes:**
- This node is a lifesaver for WAN 2.2 video projects, letting you control resolution, video length, and AI settings in one place, perfect for low-spec laptops.
- It automatically adjusts resolutions to work with the AI while keeping your desired look, and splits sampling steps for advanced workflows.
- The naming options make it easy to organize multiple video clips, and the first-generation switch helps automate setup changes for clip extensions.

#### Installation Instructions
1. Clone or download this repository to your local machine.
2. Copy the repository folder to your ComfyUI custom nodes directory: ComfyUI/custom_nodes/
3. Ensure dependencies (e.g., PyTorch, ComfyUI’s sampler libraries) are installed. These are typically included with ComfyUI.
4. Restart ComfyUI to load the `FossielCentralControl_v2` node.
5. Find the node in ComfyUI under the category `Fossiel/QoL`.

## History
2025/10/26 – Launched with Fossiel Central Control node.

## Credits
- Developed with help from Grok3
- All the developers who make tools available to everyone using local AI
- Model developers for supplying fantastic open source models, free of charge.
