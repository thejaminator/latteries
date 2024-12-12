# Slide-driven development


I am a researcher. Previously I was a MATs scholar under Ethan Perez. Now I work under Owain Evans.
Every week, I have to share research updates with my team.
Research communication through slides is an important skill that I had to learn, to update my research mentor.
I discuss what I found to be useful. To make clear slides that your mentor can understand.

I find discussions where I don't understand the experiments well unproductive. So spending time to clearly communicate the experiments is very important.
Here are some broad principles.

## Have a summary slide to set the frame
Your mentor probably manages multiple projects and people.
The first slide should recap the motivation of what you are working on. And a summary of your progress.

There are two main messages to convey. It sets the frame for what your mentor should think about.
- What I tried worked. You mentor will now think about sanity checks / control experiments . Further work e.g. in other setups.
- What I tried didn't work. You mentor will now think how to get stronger effects. e.g. training with better data / better prompting.

Ideally, place a chart conveying the main message. <Todo: Show two possible charts. Bar chart goes up and bar chart goes down.>


# Simple charts to describe research experiments
This is when you describe the experiment you did.
Always always always include your prompt!
Then, have a chart that shows the results.

Start with a simple charts. This is important! You want to convey a main message "my intervention worked" or "my intervention didn't work".
If you show a complicated chart, its hard to know what to pay attention to.

I tend to stick to bar charts. Its a universal chart that people understand.
What bars to show? This bars tend to be "model before my intervention", "control" and "model after my intervention".

As a rule of thumb, limit the number of bars to 2-3.

<show image of many bar charts>
Having too many bars makes it hard to focus attention on what your message is.


What if you had to run many setups? Show only the most important ones that you think your mentor would be interested in.
By all means, you can have more bars in a different slide at the end. Just in case your mentor wants to see more.

Show errors bars in your chart. This shows the uncertainty of your experiment.
Mentors want to know if your findings come from noise or not. As a new researcher, it takes time to build trust in your results. 
Be ready to show that you have done simple checks e.g. statistical significance based on sample sizes.

Label your axes. Is it accuracy? Is a loss? Indicate what a higher value is. Does that show that you are getting better results?


## One plot, one message
what's an example of too much information?
E.g. "Trainign on more data improves performance. But if you train on this particular dataset, it doesn't improve performance. Same this other dataset kinda helps. 
Anyways I have a new dataset that I think is better. I did a sweep over hyperparams and datasets. And now I have the best thing."
Thats is too many messages. You just want to say "training on this particular dataset worked." 
So in your main slides, show that particular set up.

# Always be ready to show your prompt
Highlight what part of your prompt to pay attention to. Saves time having to read the whole prompt.
If the prompt is long, can truncate the prompt, and put the full prompt in the appendix slides.

What are you testing on?
What are you training on?


# Think about baselines
What are some simple ways that would invalidate your results?


# Have backup slides ready
Here you have dump your "appendix slides". Predict what questions your mentor might ask. E.g. ablations, scaling with more data. Do these change the results?
Can be more "dirty". Not as polished.

# Concrete discussion points
Specific challenges guidance on.
Before your meeting, list what you think your next steps should be. Seek feedback from your mentor about whether this priorites are correct.
Any resource requests e.g. bottleneck on compute.

# Improving on your slides further
When I first started, it was hard for me to predict whether my slides are understandable. One way to improve is to ask a friend to read your slides and give you feedback.
Ask them to be honest about when they are confused. Notice when they ask clarifying questions e.g. "What do you mean by this word?". 
It means that maybe you need a slide to explain that word, incase your mentor is also confused. Even though maybe you have explained it in the previous session, you may need to remind them. So have a backup slide ready to explain that word.

# Final remarks
When I first started, I had to invest a lot of time to make slides. 1-2 days. This include things like "oh this chart does not clearly communicate my results, I need to tweak it". Eventually I got better at it so it takes half a day.
It is still a significant amount of time. But I think it is worth it. Communication is important! If you can't communicate your research, did you really research? :troll:










