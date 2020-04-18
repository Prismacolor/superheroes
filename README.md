# superheroes
Creating a data analysis project based on comic book characters.

Please note that this is by no means based on canon and should not be taken seriously, though I am sure it would spark 
some fun and rather interesting debate. This is based on rather simplified information I gleaned from the internet 
regarding these specific characters and their supposed abilities. I attempted to see if there were any clear trends 
regarding things like gender, property (DC or Marvel), abilities, and estimated power levels. 

For the purposes of this project please note that innate abilites are considered powers and abilities inherent in a 
character's body without the use of magic, items, or the need to engage in rituals in order to preserve them. For example,
Thor has innate abilities because he was born on Asgard, but Jane Foster may only have those powers when she picks up Mjolnir, 
so maybe her powers are not immediately accessible, and therefore are not considered innate.

The assignment of power levels had to be somewhat arbitrary as some characters do not have clearly defined power levels I could find. There were some clear exceptions. For example, most comic book fans know that Legion and Storm are considered Omega level, but for others, the water is a little murky. Determining power levels can also be tricky because power levels in Marvel are not necessarily defined the same way in DC, though for the sake of this project, I tried to have somewhate comparable levels. Hence Darkseid, one of DC's most powerful characters, being listed as on the same level as Marvel's Onslaught, who is stronger than the famed telepath Charles Xavier. 

At first, I was unable to find any clear trends with my data, which could be due to a multitude of factors, like: 

1) Comic book writing is extremely arbitrary and subject to change, and each writer displays powers and abilities differently. 
2) There is no real, clear power scale for either universe. 
3) There is no clear connection between abilities and power (see number 1.) Just because you're a telepath doesn't mean you're Omega.
4) My data was encoded too simply, which resulted in many data points being nearly identical, thus not presenting any clear trends. 
5) Gremlins

I did give it a second go, and added a formula that used the abilities and statuses of each characters to compute a "power score" for each person. After doing that, I made a few plots to see if there was any relationships between certain features and the new power score category. I observed that the most powerful characters in my dataset were villains, but I wonder if this would change had I included more powerful entities and gods. I also noted that as the level rose, so too did the power score, which is what we would want ideally. I also noted that the majority of my characters had innate powers, as opposed to magic or human abilities, and that they had a greater range of levels and scores than those without innate powers. In the end I created a new final plot using only levels and power scores. The data still looks rather linear but I was able to map some centroids. I finalized the project by using a silhouette score metric to evaluate how I did, it returned a 0.5426. Not too shabby for a first go!

Though I did not discover the optimal formula for Beyond Omega Levels of Power (sadface), I had fun and learned a lot of new stuff 
on the way. I got to play with the matplotlib library, learn a little about Kmeans clustering and unsupervised learning, and build 
up my debugging skills. So there's a win! Please feel free to poke around, clone the repo, and see if you can modify it for better results. Or read the data and spend endless hours wondering who would win in a battle between Thanos and Apocalypse...Either way, have fun!
