# Volunteer Description Matching

## The Task
We need you to build a system that can match volunteer descriptions based on their similarity. We have a bunch of volunteer profiles in a CSV file, and we want to be able to find the most similar volunteers to a given description.

## What You'll Need to Do

1. **First, process the data:**
   - Take in our CSV file with volunteer descriptions.
   - Clean up the text (lowercase, remove weird characters, etc.).

2. **Next, create text embeddings:**
   - Use a Hugging Face model to convert each description into a vector.
   - These vectors will represent the meaning of each description.

3. **Finally, build a similarity search:**
   - When given a new description, compare it to all the others.
   - Return the top 3 most similar volunteer descriptions.
   - Include their similarity scores.

## Example of What We Want

Say someone inputs: "Looking for volunteers skilled in graphic design to help with non-profit branding."

Your system should return something like:
```
Top 3 Matches:
1. Volunteer_ID: 2, Description: "Skilled in graphic design and interested in creating promotional materials for non-profits." (Similarity Score: 0.92)
2. Volunteer_ID: 5, Description: "Graphic designer with experience in creating logos and branding for small businesses." (Similarity Score: 0.87)
3. Volunteer_ID: 8, Description: "Creative professional with a background in visual design and marketing." (Similarity Score: 0.85)
```

## Bonus Points
If you have time, we'd love to see:
- An analysis of how different embedding models perform.
- Integration with a vector database for faster searches.
- A simple way to fine-tune the model to better understand volunteer terminology.

## What We're Looking For
- Code that works correctly.
- A solution that's reasonably fast.
- Clear explanation of your approach.
- Consideration of how this would work with larger datasets.

## Submission Guidelines

1. **Important**: Create a branch off main with the name format `firstname-lastname-submission`.
2. Implement your solution on this branch.
3. Make regular commits with clear messages.
4. When finished, push your branch and create a pull request.
5. Do not merge your PR, it will be reviewed as is.

Keep it in Python and focus on making something that's both effective and easy to understand. Good luck!
