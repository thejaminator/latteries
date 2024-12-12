# Slide-driven development

I am a researcher working under Owain Evans, having previously been a MATs scholar under Ethan Perez.
Every week, I share research updates with my team.
Research communication through slides is an important skill. I wasn't good at it at first. I'm still improving. But I've learnt a few things that have helped me!
Here, I discuss principles I've found useful for creating clear, understandable slides.



## The summary slide to set the frame
Your mentor probably manages multiple projects and people.
The first slide should recap the motivation of what you are working on and provide a clear summary of your progress.

There are two main messages to convey, which set the frame for what your mentor should think about:
- My experients worked!: Your mentor will focus on sanity checks, control experiments, and potential extensions to other setups.
- My experients didn't work: Your mentor will focus on ways to get stronger effects, like training with better data or improving the prompting.

Ideally, include a simple chart that conveys the main message - either showing improvement (upward trend) or lack of improvement (downward/flat trend) from your intervention.


## Simple charts to describe research experiments
This is where you describe the experiment you conducted.
Always include your prompt!
Then, present a chart that shows the results.

Start with simple charts. This is crucial! You want to convey a main message: "my experiment worked" or "my experiment didn't work".
If you show a complicated chart, it's hard to know what to focus on.
I tend to stick to bar charts. They are universally understood.

What bars should you show? These bars typically represent "model before", "control", and "model after".

As a rule of thumb, limit the number of bars to 2-3.



<show image of many bar charts>
Having too many bars makes it hard to focus attention on what your message is.


What if you had to run many setups? Show only the most important ones that you think your mentor would be interested in.
You can include additional bars in a separate slide at the end, in case your mentor wants to see more details.

Include error bars in your chart to indicate the uncertainty of your experiment.
Mentors want to know if your findings are statistically significant or just noise. 
Be prepared to show that you have conducted basic checks, such as statistical significance based on sample sizes.

Label your axes clearly. Is it accuracy (higher is better)? Is it loss (lower is better)? 
Indicate what a higher value represents.


## One slide, one message. But be ready for questions
Avoid overloading your slides with too much information. For example: "Training on I data improves performance. But if I train on this particular dataset, it doesn't improve performance. This other dataset kind of helps. 
Anyway, I have a new dataset that I think is better. I did a sweep over hyperparameters and datasets. And now I have the best setup."
This is too many messages. Instead, focus on a single message: "training on this particular dataset worked." 
In your main slides, highlight that specific setup.


Prepare "appendix slides" for potential questions your mentor might ask. For example, ablations, scaling with more data, and whether these factors change the results.
These slides can be less polished and more detailed.


# Always be ready to show your prompt
Highlight the key parts of your prompt to save time and avoid reading the entire prompt.
If the prompt is lengthy, truncate it and include the full version in the appendix slides.

What are you testing?
What are you training?


# Think about baselines
What are some simple ways that would invalidate your results?



# Concrete discussion points
Come prepared with specific challenges you need guidance on.
Before your meeting, list what you think your next steps should be. Seek feedback from your mentor about whether these priorities are correct.
Include any resource requests, such as if you're bottlenecked on compute access.

# Improving your slides through feedback
When starting out, it can be challenging to assess if your slides are clear and understandable. One effective way to improve is to have a friend review your slides and provide feedback.

Ask them to point out any confusing parts and pay attention when they ask clarifying questions like "What do you mean by this term?". These questions highlight where you may need additional explanation slides.

Even if you've covered certain concepts in previous meetings with your mentor, it's good practice to have backup slides ready to refresh their memory on key terms and ideas. Your mentor likely reviews many projects, so gentle reminders can be helpful.
# Final remarks
When I first started, I had to invest a lot of time to make slides. 1-2 days. This include things like "oh this chart does not clearly communicate my results, I need to tweak it". Eventually I got better at it so it takes half a day.
It is still a significant amount of time. But I think it is worth it. Communication is important! If you can't communicate your research, did you really research? :troll:










