# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class OutfitspiderItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    outfit_title = scrapy.Field()
    item_number = scrapy.Field()
    the_path = scrapy.Field()
    # num_views = scrapy.Field()
    # num_likes = scrapy.Field()
    # likes = scrapy.Field()
    image_urls = scrapy.Field()
    images = scrapy.Field()
