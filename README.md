# Heroku-Demo

ABSTRACT
The purpose of our project is to get similar items based on user searches also known as a content-based recommendation system like Amazon.com. We will be using the database provided by Amazon in JSON format from their publicly available S3 bucket. We use attributes such as product type, brand, color, etc, to fetch similar items using text-based similarity. The project uses techniques like bag-of-words, word2vec, TFIDF to feature the text into numerical vectors and eventually gauge the similarity between them. This will help us attain the goal of obtaining a similar set of items for possibly each item. Nearest Neighbor search will be used as the final algorithm to find the similarities.

The Final product is a Web app where users can choose which method and which model they would like to use. They can find items by the apparel’s Id or using an image to find similar apparel.

Data
The Data Consists of information about 1,80,000 products and each product will have multiple features such as –

1.	Title of the product
2.	Brand of the product
3.	Color of the product
4.	Type of the product
5.	Image of the apparel, etc...

Data Source: amazon.com

