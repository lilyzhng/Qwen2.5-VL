2025 August

Inspiration, my products are my data
- delivered data
Question
1. ML team internal that builds a lot of these algorithms to measure all of this
2. how do you have models and humans working together hand in hand to produce data that is better than either one of them can achieve on their own?
3. we do a lot of work building RL environments, cannot generate synthetic 
## Edwin Chen Introduction

Hi listeners, welcome back to No Priors.

Today Allad and I are here with Edwin Chen, the founder and CEO of Surge, the bootstrapped human data startup, that surpassed a billion in revenue last year and serves top tier clients like Google, OpenAI, and Anthropic. 

We talk about what highquality human data means, the role of humans as models become superhuman, benchmark hacking, why he believes in a diversity of frontier models, the scale meta not M&A deal, and why there's no ceiling on environment quality for RL or the simulated worlds that labs want to train agents in. 

Edwin, thanks for joining us.

Great. Great seeing you guys today. Surge has been really under the radar until just about now. So can you give us a little bit of color on the scale of the company and what the original founding thesis was?
## Overview of SurgeAI
So we hit over **a billion in revenue last year**. We are the **biggest human data player in the space and we're about a little over 100 people**. And our original thesis was we just really believed in the power of human data to advance AI and we just had this really big focus from the start of making sure that we had the highest quality data possible. 

#### Can you give people context for how long you've been around, how you got going, etc. 
I think again you all have accomplished an enormous amount in a short period of time and I think you've been very quiet about some of the things you've been doing. So it would be great to just get a little bit of history and when you started, how you got started and how long you've been around.

Yeah, so we've been around for five years. I think we just hit our 5-year anniversary. So we started in 2020. So before that, I can give some of the context. Before that, I used to work at Google, Facebook, and Twitter. And basically the reason we started Surge was I used to work on ML at a bunch of these big companies. And the problem I kept running into over and over again was that it really was impossible getting the data that we needed to train our models. 

So it was just this big blocker that we faced over and over again. And there was so much more that we wanted to do, even just the basic things that we wanted to do. We struggled so hard to get the data. It was really just the big blocker. 

But then simultaneously there were all these more futuristic things that we wanted to build. If we thought of the next generation AI systems, if we could barely get the data that we needed at the time, to solve just building a simple analysis classifier, if we could barely do that then how would we ever advance beyond that? So that really was the biggest problem. I can go into more of that, but that was what we faced. 

## Why SurgeAI Bootstrapped Instead of Raising Funds
#### And then you guys are also known for having bootstrapped the company versus raising a lot of external venture money or things like that. Do you want to talk about that choice in terms of going profitable early and then scaling off of that?

In terms of why we didn't raise, a big part of it was obviously just that we didn't need the money. I think we were very lucky to be profitable from the start. So we didn't need the money. It always felt weird to give up control and one of the things I've always hated about Silicon Valley is that you see so many **people raising for the sake of raising.**

I think one of the things that I often see is that a lot of founders that I know don't have some big dream of building a product that solves some idea that they really believe in. If you talk to a bunch of YC founders or whoever it is, what is their goal? It really is to tell all their friends that they raised $10 million and to show their parents they got a headline on TechCrunch. That is their goal. 

I think of my friends at Google, they often tell me, "Oh yeah, I've been at Google or Facebook for 10 years and I want to start a company." I'm like, "Okay, so what problem do you want to solve?" And they don't know. They're like, "Yeah, I just want to start something new. I'm bored." And it's weird because they can pay their own salaries for a couple months. Again, they've been at Google and Facebook for 10 years. They're not just fresh out of school. They can pay their own salaries, but the first thing they think about is just going out and raising money. 

And I've always thought it weird because they might try talking to some users and they might try building an MVP, but they just do it in this throwaway manner where the only reason they do it is to check off a box on a startup accelerator application and then they'll just pivot around these random product ideas and they happen to get a little bit of traction so that the VC DMs them. And so they spend all their time tweeting and they go to these VC dinners and it's all just so that they can show the world that they raised a big amount of money. 

And so I think raising immediately always felt silly to me. Everybody's default is to just immediately raise. But if you were to think about it from first principles, if you didn't know how Silicon Valley worked, if you didn't know that raising was a thing, why would you do that? What is money really going to solve for 90% of these startups where the founders are lucky to have some savings? 

I really think that your first instinct should be to go out and build whatever you're dreaming of. And sure, if you ever run into financial problems, then sure, think about raising money then, but don't waste all this effort and time up front when you don't even know what you'd do with it. 

Yeah, it's funny. I feel like I'm one of the **few investors that actually tries to talk people out of fundraising often**, right? I actually had a conversation today where the founder was talking about doing a raise and I'm like why? You don't have to, you can maintain control, etc. And then the flip side of it is I would actually argue outside of Silicon Valley too few people raise venture capital when the money can actually help them scale. And so I feel like in Silicon Valley there's too much and outside of Silicon Valley there's too little. So it's this interesting spread of different models that stick.

#### Q. Edwin, what would you say to founders who feel like there's some external validation necessary to especially hire a team or scale their team? 
This is a very common complaint or rationale for going and raising more capital.

I think about it in a couple ways. So I guess it depends on what you mean by external validation. In my mind, I often think about things from the perspective of are you trying to build a startup that's actually going to change the world? Do you have this big thing that you're dreaming of? And if you have this big thing that you're dreaming of, why do you care? 

Maybe the way to think about it is in Sarah's context. If you haven't—say you're a YC founder, you haven't been at Google, you haven't been at Meta, you haven't been at Twitter, you don't have this network of engineers, you're a complete unknown, you haven't worked with very many people, you're straight out of school—how do you then attract that talent? And to your point, you can tell a story of how you're going to build things or what you're going to do, but it is a harder obstacle to basically convince others to join you or for others to come on board or to have money to pay them if you don't have a long work history. So I think maybe that's the point Sarah is making.

Yeah. So I think I would differentiate between maybe two things. 
1. One is do you need the money? So first of all, there's a difference between people who are literally fresh out of school or maybe never gone to school in the first place and so maybe they don't have any savings and so they literally need some money in order to live. 
2. And then there's others who—let's assume that you don't necessarily need money because again you've been working at Google or Facebook for 10 years or 5 years, whatever it is, you have some savings. So I would say one of the questions is, the path differs depending on those two choices or those two scenarios. 

But I think one of the questions is, do you really need to go out and hire all these people? One of the things I often see—I'm curious what you guys see—but one of the things I often see is founders will tell me, okay so I'm trying to think about the first few hires I'm going to make and they're like, 
* yeah I'm going to hire a PM. 
* I'm going to hire a data scientist. Yeah, these are one of my first five to 10 hires. 
* I'm like what? This is just wild to me. I would never hire data scientists as one of the first few people in a company. And I say this because I used to be a data scientist. Data scientists are great when you want to optimize your product by 2% or 5%. But that's definitely not what you want to be doing when you start a company. You're trying to swing for 10x or 100x changes, not worrying and nitpicking about small percentage points that are just noise anyways. 
* And same with product managers. Product managers are great when your company gets big enough, but at the beginning you should be thinking yourself about what product you want to build. And your engineers should be hands-on, they should be having great ideas as well. And so product management is this weird conception that big companies have when your engineers don't have time to be in the weeds on the details and drive things themselves. It's not a role that you come up with before.

So I guess with the initial Surge team, it sounds like you had a small initial tight engineering team. You guys started building product. You were bootstrapping off of revenue. You know, at this point you're at over a billion dollars in revenue, which is amazing. 

#### Q. How do you think about the future of how you want to shape the organization, how big you want to get, the different products you're launching and introducing? What do you view as the future of Surge and how that's all going to evolve?

Before we do that, can you just explain what the—at whatever level of detail makes sense here—what the billion dollars of revenue is? Maybe how product supports the company, who your data labelers are, who your humans are, because I think there's just very little visibility into all of that.

## Explaining SurgeAI's Product
So in terms of what our product is, at the end of the day our product is our data. 

We literally deliver data to companies and that is what they use to train and evaluate their models. So imagine when you're one of these frontier labs and you want to improve your model's coding abilities. 

What we will do on our end is we will gather a lot of coding data. And so this coding data may come in different forms. 
* Maybe SFT data. We are literally writing out coding solutions or 
* maybe unit tests—these are the tests that a good piece of code must pass. 
* Maybe it's preference data where it's, okay, here are two pieces of code or here are two coding explanations, which one's better? 
* Or these might be verifiers, okay, here's a web app that I created, I want to make sure that in the top right hand of the screen there's a login button or I want to make sure that when you click this button something else happens. 

There's a bunch of different forms that this data may take, but at the end of the day what we're doing is we're delivering data that'll basically help the models improve on these capabilities. 

Very related to that is this notion of evaluating the models. You also want to know, 
is this a good coding model? 
Is it better than this other one? 
What are the errors in which this model is weak and this model is worse? 
What insights can we get from that? 

And so in addition to the data, often times we're delivering insights to our customers, we're delivering loss patterns, we're delivering failure modes. So there may be a lot of other things related to the data, but at the end it's this universe of applications or this universe around the data that we deliver, and that is our product. 

## Differentiating SurgeAI from Competitors
Yeah and maybe going back to Elad's question, maybe product isn't actually the right word here, but what's repeatable about the company or what are core capabilities that you guys have that you would say your competitors fail to meet the mark?

The way we think about the company and the way we differentiate from others is that a lot of other companies in this space are essentially just body shops. 

What they are delivering is not data. They are literally just **delivering warm bodies to companies**. And so what that means is at the end of the day they don't have any technology. And one of our fundamental beliefs is that quality 
is the most important thing at the end of the day. 
Is this high quality data? 
Is this a good coding solution? 
Is this a good unit test? 
Is this mathematical problem solved correctly? 
Is this a great poem? 

And basically a lot of companies in this space, just as a result of how things have worked out historically, have treated quality and data as commodity. One of the ways we often think about it is imagine you were trying to draw a bounding box around a car. 

Sarah, you and I, we're probably going to draw the same bounding box. Ask Hemingway and ask a second grader. Well, at the end of the day, we're all going to draw the same bounding box. There's not much difference that we can do. So there's **a very low ceiling on the bar of quality**. But then take something like writing poetry. Well, I suck at writing poetry. Hemingway is definitely going to write a much better poem than I am. Or imagine, I don't know, a VC pitch deck. You're going to write a much better pitch deck, you're going to create a much better pitch deck than I will. And so there's almost **an unlimited ceiling in this GenAI world on the type of quality that you can build**. 

And so the way we think of our product is we have a platform. We have actual technology that we're using to measure the quality that our workers or annotators are generating. If you don't have that technology, if you don't have any way of measuring it

## Measuring the Quality of SurgeAI's Output
Is the measurement through human evaluation? Is it through model-based evaluation? And I'm a little bit curious **how you create that feedback loop** since to some extent it's a little bit this question of how do you have enough evaluators to evaluate the output relative to the people generating the output, or do you use models, or how do you approach it?

I think one analogy that we often make is think about something like Google search or think about something like YouTube. You have millions of search results, you have millions of web pages, you have millions of videos. 
How do you evaluate the qualities of these videos? 
- Is this a high quality web page? 
- Is it informative or is it really spammy? 
- The way you do this is you just need—you gather so many signals. You gather page-dependent signals, you gather user-dependent signals, you gather activity-based signals, and all these feed into a giant ML at the end of the day. 

And so in the same way, we gather all these signals about our annotators, about the work that they're performing, about their activity on the site, and we just feed it into a lot of these different—we basically **have an ML team internal that builds a lot of these algorithms to measure all of this**. 

## Role of Scalable Oversight at SurgeAI

What is changing or breaking as you are scaling increasingly sophisticated annotations, right? If model quality baseline is going up every couple of months, then the expectation is it exceeds what might have been a random human at some point. As you said, can draw a bounding box into all of these different fields where we have models better than the 90th percentile at some point.

So this is actually something that we do a lot of internal research on ourselves as well. So there's basically this field of **AI alignment called scalable oversight**, which is basically this question of **how do you have models and humans working together hand in hand to produce data that is better than either one of them can achieve on their own?** 

And so even today something like writing an SAT story from scratch—even today, sure, a couple years ago we might have written that story completely from scratch ourselves. Today it's just not very efficient, right? You might start with a story that a model created and then you would edit it. You might edit it in a very substantial way. Maybe just the core of it is very vanilla, very generic, but there's just so much craft that is just inefficient for a human to do and doesn't really benefit from the human creativity and human ingenuity that we're trying to add into the response. And so you can just start with this barebones structure that you're basically just layering on top of. 

And so again, there's more sophisticated ways of thinking about scalable oversight, but just this question of how do you build the right interfaces, how do you build the right tools, how do you just combine people with AI in the right ways to make them more efficient is something that we build a lot of technology for. 

## Challenges of Building Rich RL Environments
A lot of the discussion in terms of what human data the labs want has moved to RL environments and reward models in recent months. What is hard about this or what are you guys working on here?

So we do a lot of work building RL environments and I think one of the things that people really underestimate is how complicated it is, that you can't just synthetically generate it. For example, you need a lot of tools because these are massive environments that people want.

#### Q. Can you give an example just to make it more real?

Imagine you are a salesperson. And when you are a salesperson you need to be interacting with Salesforce, you need to be getting leads through Gmail, you're going to be talking to customers in Slack, you're going to be creating Excel sheets tracking your leads, you're going to be writing Google Docs and making PowerPoint presentations to present things to customers. 

And so you want—basically these are very rich environments that are literally simulating your entire world as a salesperson, just imagine your entire world. So everything in the future is not on your desktop as well. 

Maybe you have a calendar to a meeting to meet a customer and then you want to simulate a car accident happening and you're getting notified of that. So you need to leave a little bit earlier. All these things are things that we actually want to model in these very rich RL environments. 

And so the question is how do you generate all the data that goes into this? Okay, you're going to need to generate thousands of Slack messages, hundreds of emails. You need to make sure that these are all consistent with each other. You need to make sure that, going back to my core example, you need to make sure that time is evolving in these environments and certain external events happen. How do you do all of this? And then in a way that's actually interesting and creative but also realistic and not incongruent with each other. 

There's just a lot of thought that needs to go into these environments to make sure that they're again rich, creative environments that the models can learn interesting things from. And so yeah you basically need a lot of tools and balance sophistication for great use. 

#### Q. Is there any intuition for how real or how complex is enough, or is it just there's no ceiling on the realism that is useful here or the complexity of environment that is useful here?

I think there's no ceiling. At the end of the day you just want as much **diversity and richness** as you can get because the more richness that you have, the more the models can learn from. The longer the time horizons, the more that the models can learn on and improve on. So I think there's almost an unlimited ceiling here.

## Predicting Future Needs for Training AI Models
If you were to make a five or 10 year bet on what scales most in terms of demand from people training AI models and types of data, is it RL environments or is it traces on types of expert reasoning, or what other areas do you think there's going to be a really large demand for?

I think it will be all of the above. I don't think RL environments alone will suffice just because, it depends on how you think, but there are RL environments but oftentimes these are very rich trajectories, are very long, and so it's almost inconceivable that a single reward—I think even today we often think about things in terms of **multiple rewards**, **not just a single reward**. But a thing like a single reward just may not be rich enough to capture all the work that goes into the model solving some very complicated goal. So I think it'll probably be a combination of all those.

## Role of Humans in Data Generation
If you assume eventually some form of superhuman performance across different model types relative to human experts, how do you think about the role of humans relative to data and data generation versus synthetic data or other approaches? At what point does human input sort of run out as a useful point of either feedback or data generation?

So I think **human feedback will never run out** and that's for a couple reasons. Even if I think about the landscape today, I think **people often overestimate the role of synthetic data**. I personally think synthetic data actually is very useful. We use it a ton ourselves in order to supplement what the humans do. Again, as I said earlier, there's a lot of craft that simply isn't worth a human's time. 

##### 10 million synthetic data < 1k high quality human data
But what we often find is that, for example, a lot of times customers will come to us and they'll be like, yeah, for this past 6 months I've been experimenting with synthetic data, I've gathered 10 to 20 million pieces of synthetic data. Actually, we finally realized that 99% of it just wasn't useful. And so we're trying to find right now, we're trying to curate the 5% that is useful, but we are literally going to throw out 9 million of it. 

And often times you'll find out that, yeah, actually a thousand, even a thousand pieces of high quality human data, highly curated, really high quality human data is actually more valuable than those 10 million points. So that is one thing I'll say.

Another thing I'll say is that it's almost like sometimes you need an external signal to the models. The models just think so differently from humans that you always need to make sure that they're aligned with the actual objectives that you want. 

Let me give two examples. So one example is that it's kind of funny—if sometimes if you try, so one of the frontier models, let me just say that one of them, if you go use the frontier model, it's one of the top models or one of the models everybody thinks is one of the top. If you go use it today, maybe 10% of the time when I use it, it will just output random Hindi characters and random Russian characters into one of my responses. So I'll be like, tell me about Donald Trump, tell me about Barack Obama, and just in the middle of it will just output Hindi and Russian. It's like what is this? 

And the model just isn't self-consistent enough to be aware of this. It's almost like you need an external human to tell the model that yeah, this is wrong. 

One of the things I think is a giant plague on AI is **LMSYS Arena**, and I'll skip the details for now. But I think right now people will often—it's like if you train your model on the **wrong objectives**, so the mental model that you should have of LMSYS Arena is that people are writing prompts, they'll get two responses, and they'll spend 5-10 seconds looking at their responses and they'll just pick whichever one looks better to them. So they're not evaluating whether or not the model **hallucinated**. They're not evaluating the factual accuracy and whether it followed any instructions. 

They're literally just vibing with the model and, okay, yeah, this one seemed better because it had a bunch of formatting. It had a bunch of emojis. It just looks more impressive. And people will train on basically an element of subjective and they won't realize all the consequences of it. And again, the model itself doesn't know what its objective is. It's like you almost need an external quality signal in order to tell it what the right objective should be. And if you don't have that, then the model will just go in all these crazy directions. Again, you may have seen some of the results with o1 before, but just go in all these crazy directions that mean you need these external validators. 

This also happens actually when you do different forms of protein evolution or things like that where you select a protein against a catalytic function or something else and you just randomize it and have a giant library of them, and you end up with the same thing where you have these really weird activities that you didn't anticipate actually happening. And so I sometimes think of model training as almost this odd evolutionary landscape that you're effectively evolving and selecting against and you're shaping the model into that local maxima or something. And so it's this really interesting output of anything where you're effectively evolving against a feedback signal, and depending what that feedback signal is, you just end up with these odd results. So it's interesting to see how it transfers across domains.

## Importance of Human Evaluation for Quality Data

These, you know, as you said, **5-second reaction academic benchmarks** or even non-academic industrial benchmarks are easily hacked or not the right gauge of performance against any given task. They are very popular. What is the alternative for somebody who's trying to choose the right model or understand model capability?

So the alternative that I think all the frontier labs view as the gold standard is basically human evaluation. So again, proper human evaluation where you're actually taking the time to look at the response. You're going to **fact check it**. You're going to see whether or not it followed all the instructions. You have **good taste** so you know whether or not the model has good writing quality. This concept of doing all that and spending all that time to do that as opposed to just vibing for 5 seconds, I think actually is really important because if you don't do this, you're basically just training your models on the analog of clickbait. So I think it actually is really important for model progress.

#### Q. If it's not LMSYS, how should people actually evaluate model capability for any given task?

What all the frontier labs find is that **human evals really are the gold standard**. You really need to take a lot of time to fact check these responses to verify they're following instructions. You need people with good taste to evaluate the writing quality, and so on and so on. And if you don't do this, you're basically training your models on the analog of clickbait. And so I think that really harms model progress. 

## SurgeAI's Work Toward Standardization of Human Evals
Is there work that Surge is doing in this domain of trying to standardize human eval or make it more transparent to end consumers of the API or even users?

So internally we do a lot of work actually today working with all the frontier labs to help them understand their models. So again we're constantly evaluating them. We're constantly surfacing loss areas for them to improve on and so on and so on. And so right now a lot of this is internal, but one of the things we actually want to do is external forms of this as well where we're helping educate people on, yeah, these are the different capabilities of all these models. Here these models are better at coding, here these models are better at instruction following, here these models are actually hallucinating a lot, so you shouldn't trust them as much. 

So we actually do want to start a lot of external work to help educate the broader landscape on this.

## What the Meta/ScaleAI Deal Means for SurgeAI
If we can zoom out and talk just about the larger competitive landscape and what happens with frontier models over time. What does a Meta-Scale deal mean for you guys or what do you make of it?

So I think it's kind of interesting in that we were already the number one player in the space. It's been beneficial because yeah, there were still some legacy teams using Scale—they just didn't know about us because we were still pretty under the radar. 

I think it's been beneficial because one of the things that we've always believed is that sometimes when you use these low-quality data solutions, people get burned on human data and so they had this negative experience and so then they don't want to use human data again. And so they try these other methods that are honestly just a lot slower and don't come with the right objectives, and so I think it just harms model progress overall. And so the more and more we can get all these frontier labs using high quality data, I think it actually really is beneficial for the industry as a whole. So I think overall it was a good thing to happen.

## Edwin's Underdog Pick to Catch Up to Big AI Companies
If you were to make a bet that an underdog catches up to OpenAI, Anthropic, and DeepMind, who would it be?

So I would bet on xAI. I think they're just very hungry and mission-oriented in a way that gives them a lot of really unique advantages. 

## The Future Frontier Model Landscape
I guess maybe another broader question is, do you think there's three competitive frontier models, 10 competitive frontier models a couple years from now? And is any of those open source?

Yeah. So I actually see more and more frontier models opening up over time because I actually **don't think that the models will be commodities**. I think one of the things that has actually been surprising the past couple of years is that you actually see all of their models have their own focuses that give them unique strengths. 

For example, I think Anthropic has been really amazing at coding and enterprise, and OpenAI has this big consumer focus because of ChatGPT. I actually really love Claude's model personality, and then Grok, you know, just has a different set of things that it's willing to say and to build. 

And so it's almost like every company has a different set of principles that they care about. Some will just never do one thing. Others are totally willing to do it. Others just have different—models will just have so many different facets to their personality, so many different facets to the type of skills that they will be good at. And sure, eventually AGI will maybe encompass all this, but in the meantime, you just kind of need to focus. There's only so many focuses that you can have as a company. 

And so I think that just will lead to different strengths for all the model providers. So I think today, you know, we already see a lot of people including me, we will switch between all the different models just depending on what we're doing. And so in the future I think that will just happen even more as people are just using more and more models for, or using models for different aspects of their lives, both their personal and their professional lives.

## Future Directions for SurgeAI
Going back to something Elad mentioned, where should we expect to see Surge investing over time? What do you think you guys will do a few years from now that you don't do today?

Again, I think I'm really excited about this more public research push that we're starting to have. I think it is really interesting in that a lot of the—for obvious reasons, a lot of the frontier labs they're just not publishing anymore. And as a result of that, I think it's almost like the industry has fallen into a trap that I worry about. 

So maybe to dig into some of the things I said earlier with some of the **negative incentives** of the industry and some of the concerning trends that we've seen. 

So going back to LMSYS, one of the things that we'll see is a lot of researchers, they'll tell us that their VPs make them focus on increasing their rank on LMSYS. And so I've had researchers explicitly tell me that they're okay with making their models worse at factuality, worse at following instructions as long as it improves their ranking, because their leadership just wants to see these metrics go up. And again, that is something that literally happens because the people ranking these things on LMSYS—they don't care whether the models are good at following. They don't care whether the models are emitting factual responses. 

What they care about is, okay, did this model emit a lot of emojis? Did it emit a lot of bold words? Did it have really long responses? Because that's just going to look more impressive to them. One of the things that we found is that the **easiest way to improve your rank on LMSYS Arena is literally to make your model responses longer**. And so what happens is there are a lot of companies who are trying to improve their leaderboard rank. 

So they'll see progress for 6 months because all they're doing is unwittingly making their model responses longer and adding more emojis, and they don't realize that all they're doing is training their models to produce better clickbait. And they might finally realize six months or a year later—again you may have seen some of these things in industry—but it basically means that they spent the past six months making zero progress. 

And in a similar way, I think, you know, besides LMSYS you have all these academic benchmarks and they're completely divorced from the real world. A lot of teams are focused on improving these SAT-style scores instead of real world progress. 

I'll give an example. There's a benchmark called IFEval and if you look at IFEval, so it stands for instruction following eval. If you look at IFEval, some of the instructions it's trying to check whether our models can do—it's like, "Hey, can you write an essay about Abraham Lincoln?" And every time you mention the word Abraham Lincoln, make sure that five of the letters are capitalized and all the other letters are uncapitalized. It's like what is this? 

And sometimes we'll get customers telling us, yeah, we really need to improve our score on IFEval. And what this means is again, you have all these companies or all these researchers who instead of being focused on real world progress, they're just optimizing for these silly SAT-style benchmarks. And so one of the things that we really want to do is just think about ways to educate the industry, think about ways of publishing on our own, just think about ways of **steering the industry into hopefully a better direction**. And so I think that's just one big thing that we're really excited about and could be really big in the next five years.

## What Does High Quality Data Mean?
Okay. Yeah. I mean, so Sarah brought up earlier how everybody kind of wants high quality data. What does that mean? How do you think about that? How do you generate it? Can you tell us a little bit more about your thoughts on that?

So let's say you wanted to **train a model to write an eight-line poem about the moon**. And so the way most companies think about it is, well let's just hire a bunch of people from Craigslist or through some recruiting agency and let's ask them to write poems. 

And then the way they think about quality is, well, is this a poem? Is it eight lines? Does it contain the word moon? If so, okay, yeah, I hit these three checkboxes. So yeah, sure, this is a great poem because it follows all these instructions. 

But if you think about it, the reality is you get these terrible poems. Sure, it's eight lines that has the word moon, but they feel like they're written by kids from high school. And so other companies will be like, okay, sure, **these people on Craigslist don't have any poetry experience**. 

So what I'm going to do instead is hire a bunch of people with PhDs in English literature. But this is also terrible. A lot of PhDs, they are actually not good writers or poets. If you think of Hemingway or Emily Dickinson, they definitely didn't have a PhD. I don't think they even completed college. And one of the things I will say is, yeah, I went to MIT. I think Elad, you went there too. And a lot of people I knew from MIT who graduate with a CS degree, they're terrible coders.

And so we think about quality completely differently. **What we want isn't poetry that checks some boxes** and, okay yeah, checks these boxes and uses some complicated language. We want the type of poetry that Nobel Prize laureates would write. So what you want is, okay, we want to recognize that poetry is actually really subjective and rich. 

Maybe one poem is a haiku about moonlight on water, and there's another poem that has a lot of internal rhyme and meter, and another one that focuses on the emotions behind the moon rising at night. And so you actually want to capture that there's thousands of ways to write a poem about the moon. There isn't a single correct way, and each one gives you all these different insights into language and imagery and poetry. 

And if you think about it, it's not just poetry, it's like math. There's a thousand ways probably to prove the Pythagorean theorem. And so I think the difference is that when you think about quality the wrong way, you get commodity data that optimizes for things like inter-rater agreement and again checking boxes off of some list. 

But one of the things that we try to teach all of our customers is that **high quality data actually really embraces human intelligence and creativity**. And when you train the models on this richer data, they don't just learn to follow instructions. They really learn all these deeper patterns about all the stuff that makes language in the world really compelling and meaningful. And so I think a lot of companies, they just throw humans at the problem and they think that you can get good data that way. 

But I think you really need to think about quality from first principles and what it means, and you need a lot of technology to identify, yeah, that these are amazing programs and these are creative math problems and these are games and web apps that are beautiful and fun to play, and these ones are terrible to use. So I think you really need to build a lot of technology and think about quality in the right way. Otherwise you're basically just **scaling up mediocrity**.

#### Q. That sounds very domain specific. So in every domain are you building a lens of what quality looks like along with your partners?

Yeah, I think we have holistic quality principles, but then often times there are differences per domain. So it's a combination of both.

## Conclusion

I think we got all the core topics. Nice work on podcast number two, Edwin, and thanks for doing this. Congrats on all the progress with the business.

Yeah. No, thanks so much for joining us. Yeah, it was great meeting you guys.

