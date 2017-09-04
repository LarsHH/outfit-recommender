# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import scrapy
from scrapy.contrib.pipeline.images import ImagesPipeline
from scrapy.http import Request
from scrapy.exceptions import DropItem

class MyImagesPipeline(ImagesPipeline):
    #Name download version
    def file_path(self, request, response=None, info=None):
        image_guid = request.meta['the_path']
        return 'full/%s' % (image_guid)

    #Name thumbnail version
    def thumb_path(self, request, thumb_id, response=None, info=None):
        image_guid = thumb_id + response.url.split('/')[-1]
        return 'thumbs/%s/%s.jpg' % (thumb_id, image_guid)

    def get_media_requests(self, item, info):
        #yield Request(item['images']) # Adding meta. Dunno how to put it in one line :-)
        for i in range(len(item['image_urls'])):
            # item['images'][i]['path'][-5] = str(i)
            yield Request(item['image_urls'][i], meta=item)

class OutfitspiderPipeline(object):
    def process_item(self, item, spider):
        return item
