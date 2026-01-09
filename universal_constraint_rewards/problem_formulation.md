### **Scene Representation**

Let a 3D scene be defined as an unordered set of ( N ) objects:

$$  
S = {o_1, o_2, \dots, o_N}, \quad o_i = (c_i, t_i, s_i, \theta_i)  
$$

where:

- $( c_i \in \mathbb{R}^K )$: object category one hot encoding with K classes (e.g., _chair_, _table_)
    
- $( t_i \in \mathbb{R}^3 )$: object centroid translation
    
- $( s_i \in \mathbb{R}^3 )$: object size parameters (width, height, depth)
    
- $( \theta_i \in [0, 2\pi) )$: orientation around the vertical axis
    

The object’s geometry is retrieved from a database of canonical CAD models (e.g., **3D-FUTURE**) based on its predicted size and category, following prior works such as **ATISS** and **DiffuScene**.

Thus, the generative model operates in _object-parameter space_, while full scene meshes are reconstructed via retrieval and placement.

---

### **Textual Constraints**

At inference time, the user provides a natural-language description ( T ), which specifies explicit or implicit constraints on the scene layout:

$$  
T = \text{"Place the bed near the wall and ensure the room is wheelchair accessible."}  
$$

(Note: since we use pretrained diffusion model to generate scenes not rule based procedures, obvious constraints such as no collision, static equilibrium, realistic positioning are already satisfied by generated scene, we are interested on the hard constraints that are explicitly mentioned in the text instruction. If instruction is very high level eg make the room wheelchair friendly, llm parser should be able to reason through what constraints this instruction entails eg path clearance, object accessibility from wheel chair.)
An **LLM-based parser** converts ( T ) into a structured set of constraints:

$$  
C = {c_1, c_2, \dots, c_M}  
$$

Each constraint ( c_j ) may apply to:

- **Unary terms** — constraining a single object’s position, size, or orientation
    
- **Binary terms** — describing pairwise spatial relations between objects
    
- **Global terms** — encoding holistic scene properties (e.g., accessibility, spaciousness)
    

These constraints may be _soft_ (preference-based) or _hard_ (must-satisfy).

---

### **Reward Models**

To evaluate how well a generated scene satisfies the textual constraints, we define a set of non-differentiable reward functions:

$$  
R = {R_1, R_2, \dots, R_L}, \quad R_\ell: (S, C) \rightarrow \mathbb{R}  
$$

Each reward model $(R_\ell )$ measures a specific type of constraint satisfaction, such as:

- geometric validity,
    
- relational consistency, or
    
- functional affordance.
    

The total reward is a weighted combination:

$$  
R_{\text{total}}(S, C) = \sum_{\ell=1}^L w_\ell , R_\ell(S, C)  
$$

The reward functions may depend on **spatial reasoning modules**, **affordance estimators**, or **LLM-based evaluators**, and are generally _non-differentiable_ with respect to the generative model parameters.

---

### **Objective**

Given:

- a pretrained diffusion-based scene generator $( G_\theta(z) )$ that maps a noise vector ( z ) to a scene ( S ), and
    
- a set of user constraints ( C ) with associated reward functions ( R(S, C) ),
    

the goal is to generate a scene that maximizes the overall constraint satisfaction:

$$  
S^* = \arg\max_{S \sim G_\theta} R_{\text{total}}(S, C)  
$$

If certain hard constraints are not satisfied, additional **regeneration** or **local refinement** steps may be applied until the constraint set ( C ) is met.


---
for practical purpose the input scene(unnormalized and in world coordinates) to reward model is in the shape Batch_size, number_of_furniture, concat(one_hot_encoding(number_of_furniture_type+empty_object_class_at_last), centroid(x,y,z), size(x,y,z), cos theta, sine theta). and since this is for unconditional scene generation, we also do not have the explicit bounds of the scene. only input reward fuction gets is sampled scenes of shape b,12, 30 and num_classes=22 optional arg.
- scene is a room
- y is vertical direction xz plane is floor
- angle theta is the z angle
- size is sx/2, sy/2, sz/2
- empty slot that does not have anyfurniture should have index in ohe num_classes - 1 and it has almost zero size, centroid and orietations. we ignore the empty object while calculating rewards.
    

- ignore the empty slots while calculating any reward because they are removed before rendering scenes so they do not occupy any space

- reward model should be a function my program can call passing args eg get_llm_reward()<this is what you are implementing , also put kwargs to avoid error is more than necessary args are passed to it 

- since this runs in hpc in huge batch each scene needs to be processed in parallel in cpu for efficieny because reward logic may have conditionals and loops
- while desiging reward functions, we should be mindful not to reward degenerate behaviours. eg model generating empty scenes to get spaciousness reward.
- first reason through what makes a scene maximize that reward. find constraints. how to convert them to rewards then code