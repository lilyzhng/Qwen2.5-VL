# Introduction

## Our Mission
We're focused on building AI-powered autonomy that generalizes across multiple platforms and environments.
Our goal is to automate jobs that are dirty and dangerous.

So we built up algorithms span pickup trucks, semi-trucks, and industrial vehicles. We've been driving on highways and off-road gravel roads. The market we operate in is huge — roughly four trillion dollars across commercial and public sectors.

## Kodiak's Achievements
We are truly a leader in this space. So far, we've delivered thousands of loads and driven over three million autonomous miles across the United States.

One of things at Kodiak we are most excited about is the number of paid driverless operations. What that means is that we have a fleet of vehicles operating for our customers 24/7, 365 days, without any drivers inside — with no human in the loop. 

We have 8 trucks and growing exponentially. So far, we've logged over 3,000 hours of paid, driverless operation in real-world revenue freight — what we like to call robo-trucks.

## Operating Environments and Challenges

There's supposed to be a video here — I'll give you a sense of what it's like driving in an industrial setting without anyone in the truck.

As you can see, these are gravel roads; the video is buffering a bit. It's challenging because there are no lane markings, so you don't know exactly where to drive. This is why HD maps don't scale here — you can't map these places, when it rains, you get puddles, the dynamics change you cannot categorize, you have turns are unmarked,
There are oncoming trucks with double trailers, and you have to pull over and let them pass. You'll see cones, construction zones, so lots of dynamics all of that has to be handled in real time without human being involved.

And yes, you have cow crossing, you have to be careful. And yes, we're very cautious around cattle — because cattle have owners. If you hit one, you'll get sued. It's not like hitting a rabbit or a coyote that comes in front of you. So we treat cattle almost like pedestrians.

## Platform Generalization

So we built a Kodiak driver that generalizes across vehicles platforms — track vehicles, pickup trucks, industrial trucks, and semi trucks on highway  — and across many kinds of environments: highway, industrial gravel roads, off-road, forests, even snow, dirt trials.

Sometimes we drive over grass or uneven terrain, so there isn't even a clear "road." One of the key technologies that enables this is not relying on HD maps. We trust our perception — our eyes, like humans — to determine where to drive.

## Highway Challenges

These are some examples lanes from the highway. We've seen all kinds of things — pedestrians on the highway, we actually sees people running across the highway pitch dark at night in black hoodies. So all of that happens that we have to take care of.

Weisong was there, we take multiple multiple 270 leaves,tractor drives on highway, that's tricky because the lane is narrow, for example, when you're taking a tight cloverleaf turn with a long trailer, you're hugging the outer edge of the lane. 

And when you drive from Dallas to Houston — about 200 miles — you can't predict where lanes are closed, you cannot rely on static HD maps, because construction zones pop up all the time, lanes close, new detours appear. Dynamic environments like that require perception-based autonomy.

## Off-Road and Extreme Conditions

Off-road is even trickier, much trickier than highway. Yesterday we were discussing snow. Last year, we were scheduled for an Army demo in Michigan in December. It wasn't supposed to snow, but when we get there, a week before the demo, a snowstorm hit and stayed for 4 days — six feet of snow on demo day. And it was off-road, so nobody was there to shovel the snow out. We still drove there successfully.

How did we do it? Here you can see, In industrial areas, dust clouds are common — when another truck overtakes us, it leaves a huge dusk cloud, lidar just sees a wall of points. So you don't want that to keep you stopped on the road, right? You will need to rely on other sensor modality like camera. Sometimes it is too much, you have to wait for it to pass.

This is what I was talking about, this is a large puddle, when it rains, this is what you get. Here we have a house trailer being towed on gravel road, so there are some out of distribution you have to take care of. We got snow, sun glare.

Sometimes we drive on grassy fields with no clear road, you just know where the you want to get to.

## Dynamic Environments

One thing is clear that environments are dynamic. When you talk about highway, especially when you drive on highway 101 in california. I can swear that everyday when I drive on 101, there is at least 1 accident.

Highways bring different challenges — constant accidents, messy lane markings, duplicate paint lines that confuse even humans. Construction zones, narrow bridges, cranes, vehicles towing other vehicles, and of course pedestrians or workers on the shoulder. So me as a human gets confused, but all have to be handled safely. 

If someone is stopped on the right shoulder in California, the rule (and I think it's U.S.-wide) is that you must slow down or change lanes.

We have encounter cows crossing, dense vegetation in forests, imagine that you are going through the forest. You have vegetation on both sides. Some of them might have occluded you. We happen to have 2 sensor faults on both sides. They see different things. So even one is occuluded, we have another sensor part to see. We have muddy tracks

# Technical Architecture

## Four Steps to Autonomy

And I think there are mostly four steps towards it. You need to be able to understand the real world. You have to be able to build a holistic understanding of what the vehicle is seeing. And you need to be able to model the environment and actors accurately. You need to be able to reason about how they interact with each other. For using lane lines, you have to be able to reason about which one you should follow, how other agents are interacting with you and each other. Uh, this is behavior alignment. Um, I think it's super important. Your behavior on highway is going to be very different from the behavior in a structured area. It's not only a function of the environment, but also a function of what kind of trailers you're carrying. For example, in deployment, basically, we might haul one trailer, two trailers, or three trailers. So, depending on how much weight we are carrying and how big the whole train is.

## Redundancy and Safety

If there's a big truck coming from the front, or if it's a pickup truck, our behavior is going to be different to make sure that we react safely. I think the only way to do that is to have redundancy, especially—remember that we don't have anybody in the vehicle. Um, so that means you need to have redundancy. So we unveiled our driverless platform last year, which has redundancy in every layer—we have redundant steering, we have redundant braking, we have redundant power and compute, and this thing that we call the sensor pod. It combines all the sensors, and there are multiple cameras, lidars, and radars inside the sensor pod, and then we mirror that to both sides of the truck. So, we have redundancy in the hardware as well as software. 

## Kodiak Driver Engine Architecture

Here, I'm going to give a high level, top view of our overall Kodiak Driver engine. I'm not going to go into too much detail, but here we have a Kodiak Foundation model that sort of gets pre-trained on huge corpus of data—we've driven millions of miles, and we have a lot of data now. Obviously, all of the data is not useful, and that's why you need to have an AI flywheel doing that. The AI flywheel only looks at the data that actually increases the value of the data set. Something that we call parallel modular cognitive architecture—um, within that, we have multiple neural paths that are all running in parallel, and that sort of gives us the redundancy, uh, within the system. So, if one of the neural paths doesn't work for any reason, we are still good, so we build a lot of redundancy within our system. 

## AI Flywheel and Data Pipeline

Um, I'll talk about one of those neural paths in the next slides. But yes, we have the Kodiak Foundation model and then we fine-tune many, many neural paths to be very efficient on the truck. One of the important things here, and I talked about data mining—we have this thing that we call Language and Vision Analysis engine, LAVA, that is a VLM running on board, and it continuously looks for interesting things on the road, and whenever it finds something, it tags that to create a snippet—might be of different lengths, depending on what it finds. So there's both a machine learning way of finding things on the road, and there are heuristic ways of finding things. Um, and those are the only events that we get out of our LAVA. You know these trucks have many sensors. There are like eight cameras and four lidars, multiple radars—like a terabyte an hour. There is no way you can store all of that when you are going from, let's say, California to Texas. We drive for hours and hours, and it is not possible to log, it's not possible to store. Uh, and even if you did store all of them, there is no human that can go through all of it and, like, oh, this is interesting, let me grab that. So, all of that needs to be automated. So, it's very important for you to have this LAVA engine that looks for those events, and once it finds those events, you have to pass that through some engine that adds it back to our training set. Both auto labeling and smart labeling. So, with auto labeling, things get fed through a bigger, larger model that is much better at predicting everything that you are interested in. This model should be able to give you that, and here since it runs offboard, not onboard, it should be able to do a much, much better job. And, uh, what's important is—within the truck, you only have a limited timeline, but offboard we can go forward and backward in time, and you can combine information. With auto labeling, you want to figure out what is the quality of it, and if it is something that requires human annotation, you get humans in the loop. They get labeled, and then they get fed back to our training data set. That's where we have both on-prem and on cloud infrastructure to train these models and deploy them to our trucks.

## 3D World Understanding: SensorFuse

So, I talked about 3D world understanding. How do we do this? There are many ways of doing it. I'm just going to talk about this one model that we call SensorFuse. This is our spatial temporal multimodal model.

It takes cameras, lidars, and radars, not just across space, but also across time, and it fuses them all together and makes sense of that.

We have visual encoders that are domain specific. So, for example, we have a camera encoder, a lidar encoder, and a radar encoder that extract features, and then we lift those features into a space that makes sense—that is a metric space, so that you can reason about things in the 3D world. So, 2D images, 2D features need to be converted—some people do it in bird's eye view, some people directly do it in 3D. Um, so you need to pull all of these into the metric space, and you need to take care of the ego's own motion. There has to be some reasoning about the object itself moving across time since you have observations at different times. So, if there is an object that was moving at a certain speed, you're going to observe it at different locations. So there is ego motion, and there is the object motion. Um, and then you encode all of that through a spatial temporal feature encoder, and then you can decode different tasks that you might be interested in. So we, for example, have multiple tasks—I'm showing four of these here, so you get 3D bounding boxes, you get 3D segmentation, occupancy grid segmentation. We also have something that we call the ground model, which is where you reason about the ground surface, and it's much easier to reason about on highways, much more difficult in the off-road case. And to give you some sense of it—when you're going through a forest, for example, you want to stop for all the tree trunks or the boulders or things that are solid, but you don't want to stop for soft vegetation that is maybe just hanging. So, how do you reason about what's traversable and what is not? The model takes care of that here. Um, so that is a model, but that by itself is not enough. You need to have the AI flywheel that sort of builds the loop.

## Five Major Components for End-to-End Systems

To make the end-to-end system work as intended, so I think there are five major components to it.

### Simulation and Real World Experience

Simulation is great, I think yesterday there was a great talk around the neural simulators and the procedural ways of doing things better. Uh, you know, machine learning ways of just generating scenes. It's great, and it's evolving very quickly. So I'm very bullish on where it's going to go. I am waiting for a day where we can simulate things end-to-end with all the sensors. That will be amazing. But it's not here yet. Plus, even if it was awesome, I actually think there is no replacement for real world experience, right? You need to be able to go out there, drive millions of miles, and encounter weird things—you cannot necessarily simulate everything. Well, let me say this, you can only simulate the things that you can already think of, right? If you're trying to build that and encode that yourself. But in reality, you come across many weird things. And you're, like, oh, like helicopters—a helicopter just landed on the road. Actually last year, or maybe two years ago, on I-85, people who were in the area would have realized there was a plane that crashed on the highway.

I don't know if you remember. I actually went out there, and I collected some data. Um, so I remember that there was a private plane that landed on the road. So, how are you gonna deal with that? Like, I mean, you can simulate it, but now I can think of that situation because I experienced it. I would not have thought of that before.

### Data Mining and Auto Labeling

I'll talk a little bit about that. We have a VLM way of doing that. Different companies have different ways of doing that, but it's extremely important for you to understand, uh, what scenario is not represented well in your data set, and you want to gather that. You really want to have this uniform distribution in your data set to the extent that you can, that represents all the scenarios somewhat equally well. Auto labeling is a really important part of it. All the samples that you gathered from the road, you have to feed that through auto labeling. You can also get it human labeled, but there is a scale problem here. The only way to, uh, you know, overcome that scale issue is to have algorithms that can automate a lot of it.

### Training Strategies and Deployment

I think there are two strategies—you could retrain your overall model and then post-train, or you can have a pre-trained model that could already be pre-trained on a huge corpus of data. So, every time you have this iterative improvement, you just post-train on that. And lastly, deployment. I think this is one of the most important parts of a production system. So whenever you have a new model, how do you know what its impact is on the overall system? The machine learning and computer vision community is very focused on improving the test metrics.

If you improve one model to do really well on, I don't know, one particular metric—does it make your overall product more efficient? You know, does it save you money? Does it save you time? Um, so we have end-to-end simulations from past logs that we have collected. We have like thousands and thousands of those. We have simulated ways of deploying things. We always compute end-to-end metrics on top of the model level metrics to get a sense of what we are going to see once this model is deployed. And you can apply that to any change in the entire code base. So before you deploy, you already should have some understanding of what it is that you expect from that, and then you test it out on road, so we have different stages where you can deploy that. We have daily—as we call it, that's the branch that gets built every day—we run tests on road, and then we have a candidate that's every week. Slowly, it gets promoted all the way up to where we have really a lot of confidence in the software, and that's when we call it stable and that gets released to production, uh, but we know exactly what it is.

## Recipe for Safe AI Systems

As I said, I talked about multiple things. Um, so, in my opinion, the recipe for an acceptably safe AI system—I think there are these major components. You have to have a good AI infrastructure where you can train large models. Um, I think that's super important. Obviously, Fei-Fei was saying yesterday, everybody is compute poor. Everybody is GPU poor, and talent is scarce. Um, so at Kodiak, we don't have hundreds of thousands of H100s. I wish we did. Um, but whatever you have, you have to make the best out of it, right? You have to have the right strategy for foundation models, and how do you distill the model down into a smaller model? So, all of that gets enabled by an AI infrastructure. You have to have a smart data factory, so that goes back into the AI flywheel. How do you understand when something needs to be improved? You go mine data on the road, auto label that, get humans to review it, feed it back to the model, deploy it and, again, go gather more cases. So, you basically want to find gaps within your training systems.

Uh, foundation models play a huge role here, because at the end of the day, what you want to build is sort of a system that understands the physics of the world. And one way to do that is to have a foundation model, or a lot of people call it world models. People have different definitions of these, but essentially, it's a model that understands the physics of the world and how things move, what action impacts what—how each action impacts different agents—and then you can use that to do other tasks better. The model alignment and optimization—that sort of also feeds into how efficiently you can run these models onboard.

## Simulation with Perception in the Loop

And I guess the more you're able to optimize, the more you are able to leverage out. As you talked about the importance of the testing on road and within simulation, we at Kodiak do a lot of planning simulation and a lot of perception. Simulation is still sort of—it comes from the past logs and a lot of times what you have, what we call is pose divergence.

And if your pose divergence is way too much, then you cannot trust your perception. So looking for vision projects, maybe that would be a good one—need to be able to close this loop. In the log, you were driving in, you know, on the lane. But with the simulation, you are kind of changing the lane because somebody cut in harshly, or maybe you injected that error.

What do your camera images look like? So your camera images are going to change. The image is going to change, your observations are going to change, and from that, how do you estimate the new state of the world? Feed that back, right? So this end-to-end simulation with perception in the loop area—it's very, it's very difficult.

## Generative AI for Generalization

All right, switching gears a little bit. I think generative AI is super important. It's super cool as well, but it's also very important for generalization of your models. Um, so essentially what you want to do is, let's say you have a lot of nominal scenes, and you want to generate what it will look like in, let's say, different weather conditions, different lighting conditions. Uh, that helps you generalize your model to different scenarios. And one benefit of this—if you can ensure that your content of the image does not change, then your labels transfer. Okay. So, for example, here, what we did is we went and collected data. This is from a real world environment. I think this is somewhere in California or somewhere where we are driving through vegetation. You see lots of leaves and things like that. And we then went back and we simulated what it would look like in the presence of a lot of snow or a lot of fall colored foliage. California doesn't have fall colors, but I think this place does. Yesterday I was walking out and—so you can simulate all of that. And on top of it, you can go in and vary the degree of adjustments, so you are really creating this data set, or rather multiplying your data set when your content stays the same. Uh, but you have many variations of your data.

### Embedding Analysis and Style Transfer Validation

So, I guess the real question is, once you do that, is it still realistic enough for your models? And that's where you can do a lot of embedding analysis. So, the blue points here—it's a very small set, but the blue points represent the lower dimensional projection of the embeddings that are coming from real samples. So there we have images from real-world environments. Um, and then we go in, we convert all of that such that it looks like it came from snow. And this is basically style transfer. It can look like it came from a snow scenario. These pink samples, and now we have a very small set of data that is actually collected in snow, and that was actually collected in Denver during November a couple of years ago. So when you plot that in space, you actually see that the projections overlap, right? So you have green dots that are overlapping on the pink dots, and you have a very small number of pink ones. Um, so that's where it gets you confidence that the style transfer images are very close to what you would find in reality, so if you actually were to go in and drive in the snow, um, you would actually find samples that your models might have already seen. So, if you trained on blue and pink samples here, you might generalize well to the green samples. And as a result, we were actually able to go from sunny California to Michigan snowstorm in a week. So, even without, without seeing any snow samples—we never sent our fleet to Michigan or Alaska. Actually, that's something that I was considering. Uh, we had to figure out how to send our vehicles.

Go from sunny California to Michigan snowstorm in a week, and we did a demo that the Army was really happy with. And here, as you can see, there are no lane lines, so you have to understand where to drive. The thing that we take as an input is where to go to, and we have what we call a sparse routing map, similar to Google Maps. So, if we, as a human, were driving here, we will just follow Google Maps and, like, oh, you have a turn to take in like 200 feet. You look at what you see out there, you're avoiding obstacles, you know what a good road surface is, where you should be driving—you need to go out there and drive.

## Scene Reasoning and Data Manifold

Uh, another thing that's super important is the reasoning about the scene. I think this is the last subtopic of my talk.

Uh, one thing I wanted to sort of talk about is that the data manifold, it's a spectrum. And there are a lot of things that happen out there that you perhaps cannot—can never represent in your data set. Almost impossible for you to capture 100% of what's out there, 100% of what's possible within your training set. So, at a high level, you have data samples that you do have labels for, and those might be nominal scenarios or some non-nominal scenarios that your vehicle actually experiences, and you're able to label.

Then you have certain rare samples that you can think about, like you can enumerate. Oh, what if there was an emergency vehicle that was crossing across the road? Like, you can think about certain non-nominal scenario situations, like, what if there is a person that is right there in front of the camera, or if somebody came and put their hand on top of the sensors? So these are the things that you can think about. Maybe you can anticipate that.

But there are things you cannot think about beforehand, unless those things happen in the in the real world.

But I think the data with available labels—that's easy. You collect the data, you mine those, you go in, and label.

## Zero-Shot Detection with VLMs

That is also now possible for you to go mine, thanks to vision language models, so CLIP and similar—everybody would have heard about it. Very efficiently onboard. So, just a primer—if somebody does not know about it, here you basically encode images, you encode text, and essentially, you're trying to bring them closer in the embedding space. So, trying to maximize these diagonal correlations during training.

You feed that through the same model, and we're looking for high similarity score. And whenever you do have that, you think you have captured that. So, that actually helps you do zero-shot detection of a lot of situations. For example, here we have camera occlusions. Now, you can already think about that, and it's very hard to capture data like this. Maybe you might see this 0.001 percent of the time when somebody is maybe cleaning a camera or doing something. Um, so you want to capture these things, and you want to detect these things in zero-shot. Here's another, another one that is a person very, very close to the camera. Uh, it's very hard to put a 3D bounding box around this person that close to the camera. It's very hard to reason about. Uh, but with zero-shot prediction, you can capture some of these scenarios—just to capture the data and label them, but these are scenarios where you might want to do something different if you know there's a person very close. Um, slam on the brakes.

It works. Yeah, if you can enumerate things that you are interested in. Here you were actually interested in occlusion—the person on the bicycle, you can understand that, and yeah, and to test this. Important one. This is actually something that you—so here you can see how the sun spots on the trees, and you see when the camera gets covered with leaves, you have very high score there. So you know the cameras are occluded. The occlusion is easy to reason about for the camera. Here you have a bunch of rocks and debris on the road ahead, and you can identify that.

## High-Level vs Low-Level Scene Reasoning

Um, another way to reason about the scene, and I'm going to sort of talk a little bit about low level reasoning and high level scene reasoning. Uh, and this goes way beyond the VLM that we're talking about previously. So when you look at this particular scene—as a human, you can reason a little bit about that. You gather your context around it so—looks like there has been an accident here. There is an overturned vehicle, and if you can see it. Um, so that's really out of distribution. You have not seen a vehicle that's flipped. I think it flipped sideways, and there is a vehicle that is trying to recover it. There are some cranes, so if I'm driving through this and there are no cranes, and then I can imagine there might be recovery operations. There is a pile of dirt there.

You have a lot of things here that we, as a human, understand. Now in traditional perception sense, we likely do reason about what—what we would basically extract are detections, tracks, and maps, and obstacles. And there, we don't actually have any of these contexts. So, maybe we see some debris out there, and we see some vehicle tracks on the left. Yeah, it's fine. You come across that all the time, so you would probably pass through this, um, without thinking much. But in reality, there are people there that might jump in front of you. Um, because it's somewhat of a random—like, there is some randomness involved here, and you cannot necessarily predict what's going to happen here, especially when you are hauling, let's say, two trailers behind you. It's very hard for you to come to a stop, uh, immediately, so you have to sort of, uh, think about that. So, the traditional perception system that you are using—detecting tracks, obstacles, maps—that's what I call the low level reasoning. We as humans do high level reasoning, so we build all of this context around, and we understand—oh, there was an accident, it's a complex situation. So what we do in this situation is, um, we actually want our vehicle to send a call for help, and that's a term that we use. Now, remember that there is nobody inside the truck, and there's nobody watching remotely, so the truck has to decide when it's a complex situation. Um, so in this case, for example, we want the truck to send a call for help for a human to look at it remotely. Like, okay, this was a good call. I'm gonna help my truck here. I'm gonna have it pass cautiously, and then I'm gonna hand back the control after it goes through. That's also called the top-down approach, which is exactly what safety drivers do. An autonomous truck or autonomous vehicle, when deployed, has a safety driver.

They watch what the vehicle is doing, and in situations like this, maybe they'll let it do it by itself, but they will be ready to hit a button if something goes wrong, right? So there is somebody that is overseeing.

So the truck has to do that top-down approach.

## AI Safety Monitoring Agent

And this is exactly what we do. We build what we call an AI safety monitoring agent. And this is where you have a really large VLM that is keeping a watch across a large fleet of vehicles. So here, you could imagine, um, running something like Gemini if you were to use cloud services, or you could host your own services for DeepSeek or any open source model really. Now, they are improving every single day. But these are the models that can really reason about the scene—they have gotten so good that they truly understand the context. So, now, if you have an AI system that is sort of offloaded, but keeping an eye on many, many fleets of vehicles, doing that high level scene reasoning, it can call for help. You can think of it like this—if the AI system looks at it, it's like, oh, this is a complex scene, I think a human needs to be involved. It sends a call for help, and then somebody gets involved.

What I call is the AI Stack 2.0. It's where we have a foundational layer and you build Kodiak AI Services as an abstraction. Then we have an application layer, so the foundation layer could involve many foundational VLMs.

You define what the goal is, and then that's what this large AI system is running in the back.

## Edge Cases and Complex Scenarios

Here are some interesting scenarios, uh, that we found—a very, very small set. We're identifying many on the ground. So, here we have a vehicle or a bunch of vehicles that are kicking off dust in front of us. It's very hard to reason about, so it might wait there, let the dust settle and then go forward. Lot of animals. You have a coyote that is coming in front of the vehicle and blocking the road. Uh, we've actually seen, especially when we operate in Permian Basin—there are lots of trucks passing by, lots of people. They see this truck, they get curious. They want to get in and figure out what's going on. So, here we have somebody who came in front, parked, and tried to open the door. Now, obviously, it was locked—they couldn't get anything. Other scenarios. You have sandstorm and rainstorms.

Flooded roads. Here, we have cows that are crossing the road casually.

# Key Takeaways

All right, so I want to end with some takeaways. The very first one is a lot of people talk about end-to-end AI systems. And when they say that, a lot of us sort of understand that it's a monolithic AI system.

But it's hard to actually reason about what's going on within the scene. It's hard to make a safety case. It's hard to guarantee that you will always be safe. So, uh, building a system that all it does is lane following, and you have to have this interface going on, and you should be ready to hit the brakes—I think that system works perfectly. You can drive on highways with somebody always watching. But when you build a system where the system is responsible for the safety 100% of the time, that just doesn't work because you cannot calculate all the edge cases, um, especially when you have hundreds of thousands of scenarios.

Think about where it has come, I think.

Uh, I think today vision language models, the large visual language models tend to be really large. So, I hope that through research around that, they can bring the size down, bring the inference latency down and make it more efficient.

I talked a little bit about the AI flywheel. Super important for you to have the end-to-end training, uh, delivery loop. To have the loop closed, right, where you have trucks operating in the field—they're really operating in the field, gathering the right amount of data, right kind of data, and all of that.

And lastly, as you scale—that's the phase that we are in—you go from one truck to ten trucks to thousand trucks.

Things that were rare sort of become nominal because you come across them more often. Initially, you come across once a month. Then, you start to come across them constantly. So, we have to deal with them as if they are nominal.

Those are all the things I have to share. Now, I see a lot of people who didn't know about Kodiak. So, if you want to learn more, connect with me. I also see a lot of students here. So, if you want to join Kodiak, we're doing a lot of interesting work. Connect with me.

We are working on foundation models, generative AI, vision language models.

# Q&A Session

So, I have to say, this actually is the best talk on autonomous vehicles I have ever seen, even in our lab. We even wrote a book on this, so this—we should have invited him to give a talk in our class. You know, he's very good at planning and the algorithms, but nothing like—I go to a CVPR, I'm completely lost. I was just—the mathematics is also great on the system side, right? So that you—you bring the system, you know, together.

## Question: Snow Dynamics

A question about snow. Like, if you're from a snowy place, one thing you notice very quickly is on the first day of snow, mostly people from out of town—yeah, and like, I'm Canadian, so Montreal drivers are always very amused, because in New York, they get one millimeter of snow and the whole city jams up, so snow is like more than a perception—more than a vehicle control phenomenon, right? I was, just wondering if you wanted to comment on how to deal with the very different dynamics of snow? Which I don't think you're simulating?

**Answer:**

That's why having somewhat of a modular approach where the systems talk to each other is also important because you can do things, so it's possible for you to simulate the dynamics of snow into your control system before you actually experience them. Now, there is going to be some sim-to-real gap there.

I guess, uh, for the perception system, that's where you can use it—use generative AI to do that and bring it as close as possible to reality, right? I, I drove in Michigan for like four years, and I've seen this happen a lot, especially on highways, where, uh, you know, you're like, uh, all of a sudden, you have a snow blizzard, and you are driving on the road and you see cars flying all over the place.

It's important for us to think about it and sort of separate our perception from the rest of the system, because the rest of the system—the dynamics simulation has gotten into a place where, now, you can even simulate fluid dynamics, right? How fluids move. I think I was at ICCV a couple weeks earlier. There was a lot of good work in even simulating how smoke rolls out and how smoke behaves and all of that.

That's possible. That's how we were able to generalize, but I would say if you have more time, you could actually go out there, drive and sort of tune your algorithm based on the traction. One important distinction here is that this was off-road.

Not slippery conditions because you don't have like ice—a sheet of ice on the road. On the paved road that becomes really dangerous. And if you're going at 70 miles an hour in Michigan, people go at 95, even on the snowy days. No, no, I lived there, so I'm not driving that fast. Like, I didn't do that. But yeah, I would come across a lot of people who would do that and end up in an accident.

## Question: Spatial Temporal Context and Snow

I have a follow-up, so just in terms of the—you mentioned spatial temporal context, right? One side of—like, in Virginia, you're going on one side of a hill, you get snow. On the other side, it's all melted, right?

So the wheels and the friction, traction—all of that control loop versus this context, because there's no context switching. It's, like, oh, it's sunny, it's fine. You go around the curve and suddenly you get a snow ice patch, right?

Does the neural network do all of that, or is it more abstracted like, we are now in snow, there's still a chance of snow? I mean, when I'm driving, I monitor the temperature very precisely, like if it drops below 32, you know, change of behavior, right?

**Answer:**

So, actually, we don't nominally operate in snow. There are a few times where we have, and we're able to generalize to that, and we knew exactly what we're getting into, because it was going to be off-road.

Like that video. There is going to be snow on the road, so we talked about spatial temporal model. I think what that does is something that's completely different. Um, it's essentially trying to put together the context around the scene, and uh, given what the models are seeing—so you have three bounding boxes and what does the surface look like? We have a ditch. How does that—the model captures all of that.

Does it capture environmental characteristics?

It does, but not snow specifically, if that's what you're asking. It does like, oh, this is a road surface. It's a paved road. These are vegetation. This is a tree. This is a metal object.

In Texas. So, that's where we have our operational hub. So, we operate out of Houston and some other places. Not in New York, yeah.

## Question: Domain Transfer and Embeddings

Okay, for the second time, so we'll just take one more question. You showed one slide where you're doing the domain transfer, using embeddings to see how good the transfer is. How do you know you are looking at the right embedding?

Because in this embedding space, the transfer domain—the transferred data, plus the real data—now, they are very close to each other, but you may be looking at a completely different domain, like embedding space than what you should really be looking at.

Just because they show up in some embedding space close to each other doesn't mean you've done a good style transfer.

**Answer:**

So you need to be able to do that in the exact model that you're using onboard, right? That's when you are actually measuring—you're measuring for the right distance, you're visualizing the right thing.

If you're using embeddings from a CLIP model, the comparisons are going to be very different. What we do is, for example, the spatial temporal multimodal model that I talked about—that itself generates embeddings, and that's what we look at. So, we sort of—we sort of peek into what this model, how this model has seen different inputs in a high dimensional space. Does that make sense, right? So you're expected to do that.

Yeah. You know, before we go to—we were a little bit slightly, maybe 10 minutes behind the schedule because we started a little late.
