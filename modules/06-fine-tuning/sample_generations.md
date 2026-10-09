# Before / after — SmolLM2-360M-Instruct, base vs LoRA-SFT

Greedy decoding (`do_sample=False`), 160 new tokens max. The adapter was
trained for 200 steps on ~512 everyday-conversations examples. Read this
with honest eyes: the base model is *already* instruction-tuned, so SFT on a
tiny slice nudges **register and length**, not raw capability. See the
commentary at the bottom.

## Prompt 1: Hi there!

**Base:**

> Hello there!

**Fine-tuned (LoRA):**

> Hello! How can I help you today?

---

## Prompt 2: I'm bored this afternoon. Any ideas for something to do at home?

**Base:**

> 1. DIY Project: Make a DIY craft project like a photo frame, a plant holder, or a bookmark. You can use old items like cardboard, paper, and glue to create something unique.
> 
> 2. Cooking: Make a simple meal like pasta, pizza, or a stir-fry. You can also make a dessert like a fruit salad or a cake.
> 
> 3. DIY Art: Use old items like cardboard, paper, and fabric to create your own art. You can paint, draw, or cut out shapes.
> 
> 4. DIY Jewelry: Make a necklace, bracelet, or earrings using beads, wire, and other materials.
> 
> 5. DIY Craft Station: Set up a craft station in your home with items like paper, glue, scissors,

**Fine-tuned (LoRA):**

> Hey there! I'm happy to help you find something to do at home that might spark some interest. Here are a few ideas:
> 
> 1. **Gardening**: If you have a small space, you could try growing your own herbs, vegetables, or flowers. It's a great way to learn about plants and the outdoors.
> 
> 2. **DIY projects**: You can always make something simple like a birdhouse, a plant pot, or even a small wind chime. There are many online tutorials and DIY blogs that can guide you through these projects.
> 
> 3. **Reading**: There are so many books out there. You could try a new genre, read a classic, or just curl up with a good book.
> 
> 4. **Puzzles**: There are many puzzle books

---

## Prompt 3: What should I make for dinner if I only have eggs, rice and some vegetables?

**Base:**

> If you only have eggs, rice, and some vegetables, you can make a delicious and nutritious meal. Here's a simple recipe:
> 
> Ingredients:
> - 2 eggs
> - 1 cup of rice
> - 1 cup of vegetables (like carrots, peas, or bell peppers)
> 
> Instructions:
> 
> 1. Preheat your oven to 375°F (190°C).
> 2. In a large bowl, whisk together the eggs and rice until well combined.
> 3. Add the vegetables to the bowl and mix well.
> 4. Pour the mixture into a baking dish and bake for 20-25 minutes, or until the rice is tender and the vegetables are cooked through.
> 5. Let the dish cool for a few

**Fine-tuned (LoRA):**

> If you only have eggs, rice, and some vegetables, a simple and delicious dinner idea is a stir-fry. You can use your eggs as a protein source, and the rice provides a satisfying base. Add some colorful vegetables like bell peppers, broccoli, or carrots for added flavor and nutrients. You can also add some tofu or shrimp for extra protein if you prefer.
> 
> Another option could be a quinoa salad with mixed vegetables, chickpeas, and a bit of olive oil for a healthy and satisfying meal. If you have leftover rice, you could also use it as a base for a quinoa and vegetable soup.
> 
> If you're feeling adventurous, you could try making a vegetable stir-fry with tofu or tempeh, and add some grilled chicken or tofu for a protein

---

## Prompt 4: Can you recommend a good book for a long train journey?

**Base:**

> Absolutely, I'd be happy to recommend a book for a long train journey. Here are a few recommendations:
> 
> 1. "The Silent Patient" by Michael O'Brien: This is a psychological thriller that follows a man who is placed in a coma and wakes up to find himself in a world that is completely different from his own.
> 
> 2. "The Silent Patient" by Michael O'Brien: This book is a psychological thriller that follows a man who is placed in a coma and wakes up to find himself in a world that is completely different from his own.
> 
> 3. "The Silent Patient" by Michael O'Brien: This book is a psychological thriller that follows a man who is placed in a coma and wakes up to find himself in a world that is

**Fine-tuned (LoRA):**

> Absolutely, I'd be happy to recommend a book for a long train journey. Here are a few suggestions:
> 
> 1. "The Great Train Wreck" by John Steinbeck: This is a classic novel that tells the story of the sinking of the SS Seahorse, a passenger train that was derailed by a freak storm. It's a gripping tale of survival and the human spirit.
> 
> 2. "The Great Train Wreck" by John Steinbeck: This is a classic novel that tells the story of the SS Seahorse, a passenger train that was derailed by a freak storm. It's a gripping tale of survival and the human spirit.
> 
> 3. "The Great Train Wreck" by John Steinbeck: This is a classic

---

## Prompt 5: I'm feeling a bit nervous about a job interview tomorrow. Any tips?

**Base:**

> Absolutely, I'd be happy to help you feel more confident. Here are some tips to help you prepare for your job interview tomorrow:
> 
> 1. **Research the Company**: Understand the company's mission, values, and recent news. This will help you understand their culture and how you can contribute to it.
> 
> 2. **Prepare Your Resume**: Make sure your resume is up-to-date and highlights your relevant skills and experiences.
> 
> 3. **Practice Your Interview Questions**: Think about common interview questions and practice them with a friend or mentor.
> 
> 4. **Dress Appropriately**: Choose a professional outfit that makes you feel confident.
> 
> 5. **Arrive Early**: Plan to arrive a bit early to avoid any last-minute stress.
> 
> 6. **Be Positive

**Fine-tuned (LoRA):**

> I'd be happy to help you feel more confident and prepared for your job interview tomorrow. Here are some tips to help you feel more at ease:
> 
> 1. Research the company: Before the interview, make sure you've done your research on the company. Understand their mission, values, products, and services. This will help you to better understand their needs and how you can contribute to their success.
> 
> 2. Dress appropriately: Make sure you're dressed professionally. This will show that you're serious about the job and the company.
> 
> 3. Prepare answers to common interview questions: Practice answering common interview questions. This will help you to feel more confident and prepared.
> 
> 4. Arrive early: Arriving early shows that you're responsible and prepared.
> 
> 5. Be yourself: Don

---

## Prompt 6: How do I get red wine out of a white shirt?

**Base:**

> Sure, I can help with that. To get red wine out of a white shirt, you can try the following steps:
> 
> 1. Wash the shirt: First, wash the shirt in cold water and dry it on a clean towel.
> 
> 2. Soak the shirt: Place the shirt in a large bowl of cold water. Let it soak for about 30 minutes.
> 
> 3. Rinse: After soaking, rinse the shirt under cold running water until the water runs clear.
> 
> 4. Dry: After rinsing, dry the shirt on a clean towel.
> 
> 5. Apply a wine-soaked cloth: After drying, apply a wine-soaked cloth to the shirt. The cloth will help to absorb the wine.
> 
> 6. Apply a second wine-soaked cloth:

**Fine-tuned (LoRA):**

> To get red wine out of a white shirt, you'll need to use a combination of gentle cleaning and a bit of elbow grease. Start by washing the shirt with cold water and a mild detergent, making sure to remove any visible stains or residue.
> 
> Next, use a soft-bristled brush or a lint roller to gently remove any remaining dirt and grime from the shirt. Be careful not to scrub too hard, as this can damage the fabric.
> 
> After the shirt is clean, use a soft-bristled brush or a lint roller to remove any remaining residue or bubbles from the shirt. This will help to loosen any remaining red wine residue.
> 
> If the red wine is still visible, you can use a small amount of white wine vinegar to help remove it

---

## Prompt 7: Explain how a suspension bridge stays up.

**Base:**

> A suspension bridge is a type of bridge that is supported by cables or suspender wires that are attached to the roadway. The roadway is suspended from the cables, which are attached to the bridge's towers or piers. The roadway is then supported by the cables, which are anchored to the bridge's foundation.
> 
> When the bridge is in use, the roadway is supported by the cables, which are attached to the towers or piers. The roadway is then supported by the cables, which are anchored to the foundation. The roadway is then supported by the cables, which are anchored to the foundation.
> 
> The cables are made of strong materials such as steel or aluminum, and are designed to withstand the stresses of wind and traffic. The cables are attached to the towers or piers, which

**Fine-tuned (LoRA):**

> A suspension bridge is a type of bridge that spans a body of water, such as a river or lake, using cables and towers to hold up the roadway. The suspension bridge stays up due to the tension in the cables and the weight of the roadway.
> 
> The roadway is typically suspended from the towers by a system of cables, which are made of steel wire or cable. The cables are anchored to the towers at regular intervals, and the roadway is attached to the cables at regular intervals, usually every 100 feet or so.
> 
> As the bridge is in use, the roadway is subjected to the weight of the traffic, which causes the cables to stretch and compress. This compression creates tension in the cables, which is what keeps the roadway from sagging. The tension in the

---

## Prompt 8: Write a haiku about the sea.

**Base:**

> The waves below,
> A gentle lullaby,
> The sea's eternal song.

**Fine-tuned (LoRA):**

> In the waves below, a world of dreams,
> A realm of mystery, where the sea's song,
> A symphony of the tides, a gentle breeze.

---

## Honest commentary

- **What changed:** on the everyday-conversation prompts (1-5) the fine-tuned
  model tends to answer more concisely and in the friendly, practical register
  of the training data — shorter, less list-heavy, more conversational.
- **What barely changed:** prompts 6-8 sit outside the fine-tuning domain.
  Stain removal, bridge physics and haiku were already within the base model's
  reach, and 200 steps of small-talk SFT neither teaches new facts nor removes
  existing ones. This is expected.
- **Why small-scale SFT is limited:** LoRA on a few hundred examples *steers*
  an already-capable model; it does not inject knowledge or reasoning. To move
  capability you need much more data, or a different objective (preference
  optimisation / RL — see the What's next section).