import pandas as pd
from faker import Faker
import datetime
import numpy as np

fake = Faker()

book_names = [
    "The Night Circus by Erin Morgenstern",
    "Educated: A Memoir by Tara Westover",
    "The Silent Patient by Alex Michaelides",
    "Becoming by Michelle Obama",
    "Where the Crawdads Sing by Delia Owens",
    "The Alchemist by Paulo Coelho",
    "Little Fires Everywhere by Celeste Ng",
    "Normal People by Sally Rooney",
    "The Girl on the Train by Paula Hawkins",
    "Circe by Madeline Miller",
    "Sapiens: A Brief History of Humankind by Yuval Noah Harari",
    "The Tattooist of Auschwitz by Heather Morris",
    "The Testaments by Margaret Atwood",
    "Eleanor Oliphant Is Completely Fine by Gail Honeyman",
    "A Song of Ice and Fire series by George R.R. Martin",
    "The Subtle Art of Not Giving a F*ck by Mark Manson",
    "To All the Boys I've Loved Before by Jenny Han",
    "The Seven Husbands of Evelyn Hugo by Taylor Jenkins Reid",
    "The Goldfinch by Donna Tartt",
    "The Book Thief by Markus Zusak",
    "Witcher",
    "Lord of The Rings",
    "Winnie the Pooh"
]

stationery_items = [
    "Pilot G2 Gel Pen",
    "Moleskine Classic Notebook",
    "Sharpie Permanent Markers",
    "Staedtler Triplus Fineliner Pens",
    "Post-it Notes",
    "Zebra F-301 Ballpoint Pen",
    "Leuchtturm1917 Hardcover Journal",
    "Paper Mate Flair Felt Tip Pens",
    "Rhodia Dot Grid Notebook",
    "Uni-ball Signo Gel Pen",
    "Tombow Dual Brush Pens",
    "Midori Traveler's Notebook",
    "Faber-Castell PITT Artist Pens",
    "Field Notes Memo Books",
    "Lamy Safari Fountain Pen",
    "Sakura Pigma Micron Pens",
    "Clairefontaine Classic Notebooks",
    "Pentel EnerGel Pens",
    "Blackwing Palomino Pencils",
    "Rhodia Meeting Book"
]


delivery_reviews = [
    "Fast delivery and well-packaged.",
    "Delayed shipping but excellent customer service.",
    "The book arrived damaged, but the issue was quickly resolved.",
    "Impressed with the quick response from customer support.",
    "Smooth ordering process and prompt delivery.",
    "Received the wrong book, but they sent the correct one right away.",
    "Efficient service and hassle-free returns.",
    "The package was left outside and got wet in the rain.",
    "Poor communication regarding delivery status.",
    "Book was out of stock after ordering, disappointing experience.",
    "Received a damaged book, but no response from customer service.",
    "Bookshop staff were helpful and friendly.",
    "Delivery took longer than expected, but understandable due to current circumstances.",
    "Excellent packaging to protect the book.",
    "The delivery driver was rude and unprofessional.",
    "Shop website was easy to navigate.",
    "Book was missing from the package, still waiting for resolution.",
    "Received the book with a torn cover, disappointing.",
    "Smooth transaction and fast shipping.",
    "Shop sent the wrong edition of the book.",
]

stationery_reviews = [
    "The pen arrived quickly, very happy with the service.",
    "Ordered multiple pens, but only one arrived.",
    "The stationery section has a wide variety of products.",
    "Had to contact customer service multiple times for an update on delivery.",
    "Pen was faulty upon arrival, disappointing.",
    "Great customer service, they resolved the issue promptly.",
    "Package arrived damaged, but the contents were intact.",
    "The stationery store has competitive prices.",
    "Received the wrong color pen, waiting for a replacement.",
    "Easy returns process for defective items.",
    "Stationery arrived well-packaged and undamaged.",
    "Pen was missing from the order, still waiting for resolution.",
    "Smooth ordering process and fast delivery.",
    "The ink cartridge was dried up upon arrival.",
    "Prompt response from customer service regarding delivery concerns.",
    "Received the wrong brand of pen, disappointed.",
    "Hassle-free shopping experience.",
    "Shop website could use improvement in user interface.",
    "Ordered a set of pens, but one was missing.",
    "Pen quality exceeded expectations.",
]

# Generate purchase_date, product_name, amount_paid, review_score, and review_description for 50 records
data = []
np.random.seed(42)
for _ in range(2000): # books
    purchase_date = fake.date_time_between(start_date='-1y', end_date='now')
    product_name = np.random.choice(book_names)
    amount_paid = round(np.random.uniform(5, 100), 2)
    review_score = np.random.randint(1, 6)
    review_description = np.random.choice(delivery_reviews)
    customer_age = np.random.randint(18, 42)
    customer_sex = np.random.choice(['m', 'k'])
    data.append([purchase_date, product_name, amount_paid, review_score, review_description, customer_age, customer_sex])

for _ in range(2000): # stationery
    purchase_date = fake.date_time_between(start_date='-1y', end_date='now')
    product_name = np.random.choice(stationery_items)
    amount_paid = round(np.random.uniform(5, 100), 2)
    review_score = np.random.randint(1, 6)
    review_description = np.random.choice(stationery_reviews)
    customer_age = np.random.randint(26, 67)
    customer_sex = np.random.choice(['m', 'k'])
    data.append([purchase_date, product_name, amount_paid, review_score, review_description, customer_age, customer_sex])


# Create DataFrame
df = pd.DataFrame(data, columns=['date', 'product', 'price', 'score', 'review', 'age', 'sex'])

# Display the DataFrame
df.to_csv('sample_data.csv', index = False)