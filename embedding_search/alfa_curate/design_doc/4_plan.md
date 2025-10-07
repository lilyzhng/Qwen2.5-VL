# PT-1756 ALFA Curate: Development Plan

Authors: Lily Zhang  
Date: Oct 6, 2025

# 4. Development Plan

### 4.1 Dependency

### Applications / Active learning strategies

* Replace current strategy for obstruction scenarios?  
  * Currently using CLIP, which tagged all night scene as obstruction scenes

### Questions

* What is the priority? What matters the most for you?  
* Whether the data curation runs as an automated process or is based on manual queries  
* Whether to look purely in embedding space or use more statistical approaches  
* How to best structure the overall data pipeline architecture

**Automate Data Curation**

* Automatic queries generation, can pull metrics from bigquery, rank slices based on ped detection in the front frustum   
* Started to log metrics, can log more including training loss, use VLM agent to check loss changes over the time.

### 4.2 Productionization
