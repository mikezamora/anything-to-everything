# Feature Ideas

- Better job management tools
- Resumable jobs (that save artifacts and checkpoints at the end of every step and track overrall progress) This is also a prerequisite to expanding to distributed processing on multiple instances of anything-to-everything workers
- Track on going job progress and output current step and job details

- Add Audiobook artwork to m4b files

- Add output explorer to view and download previous job outputs

- strip unknown tokens 

```    
anything-to-everything  | >> starting inference...
anything-to-everything  |   >> Warning: input text contains 7 unknown tokens (id=2):      
anything-to-everything  |      Tokens which can't be encoded:  ['=', '=', '=', '=', '=', '=', '=']
```