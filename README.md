# Master thesis
#### Transfer Reinforcement Learning in a distributed laser-based manufacturing system. <br>
Laser manufacturing systems are often presented as the combination of multiple commonly used components that play significant roles at each step of the production. These components are the laser scanner, sensors to ensure repeatability to evaluate the quality, software to create the laser process content, movable stages to position and transport the workpiece, and others depending on the domain and purposes. Control of such a multi-system requires manual configurations of how these components interact with each other to complete the job. It inspires to seek for automation to reduce the dimensionality and complexity of the process and eliminate a tedious manual work of adjusting the system with every new component in it. Instead of manually programming every single laser machine for a specific process we propose to introuce the Reinforcement learning approach for automation purposes.

### Description
#### Research: <br> 
show applicability and usability of RL to control the specific set of hardware devices according to the laser job specification to achieve a certain laser job results. <br>
#### Simple words: <br>
given: set of hardware components, laser process specifications (efficient, high-quality, etc.), laser task (trivial: shoot on dots) <bt>
do: learn how to complete the task with all what is given.
#### Approach: <br>
Four steps automation:
- Encapsulate every hardware component into a separate container to build a **microservice architecture** for easier manupulation pf different hardware and scaling. 
- Introduce **Reinforcement Learning (RL)** into such a system. Choose a simple setup sonsisting of a camera and a laser scanner and solve a very trivial task, since the focus is on the compexity of the system (task: position the laser beam on the target location).
- Expand the system with an additional component (movable stages) and accelerate learning of a task by leveraging the knowledge from already accomplished problems in the previous setup -> use **Transfer Learning** in RL (used 2 methods in this work: *direct Q-values* transfer and *Probabilistic Policy Reuse* for results's comparison)/
- Assess Transfer Learning in RL to accelerate learning in a slighlty changed setup (ex.:camera shift to the right) with the knowledge from the past domains. 



