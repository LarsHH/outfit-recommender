from outfitspider.items import OutfitspiderItem
import datetime
import scrapy
import re


class OutfitSpider(scrapy.Spider):
    """
    Parses outfits from Polyvore.com
    """
    name = "my-outfit-spider"

    def start_requests(self):
        """
        Entry point for the spider.

        The url contains links. When done with this url we get
        the next set of items by modifying the url and requesting
        it.

        """
        token = 30
        for date in [None, 'day', 'week', 'month', '3m']:
            for token in range(30, 900, 30):
                if date is not None:
                    url = 'https://www.polyvore.com/cgi/search.sets?.in=json&.out=jsonx&request=%7B%22item_count.to%22%3A%2210%22%2C%22date%22%3A%22{}%22%2C%22item_count.from%22%3A%224%22%2C%22page%22%3Anull%2C%22.passback%22%3A%7B%22next_token%22%3A%7B%22limit%22%3A%2230%22%2C%22start%22%3A{}%7D%7D%7D'.format(date, token)
                else:
                    url = 'https://www.polyvore.com/cgi/search.sets?.in=json&.out=jsonx&request=%7B%22item_count.to%22%3A%2210%22%2C%22item_count.from%22%3A%224%22%2C%22page%22%3Anull%2C%22.passback%22%3A%7B%22next_token%22%3A%7B%22limit%22%3A%2230%22%2C%22start%22%3A{}%7D%7D%7D'.format(token)
                yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        """
        Parses a set of outfits and yields links to each individual
        outfit page.

        Args:
            response (Response): The webpage containing sets of outfits
        """
        t = response.text
        links=re.findall('<div class=\\\\"title\\\\">\\\\n            <a href=\\\\"..([\w\d?=/]*)', t)
        for outfit_url in links:
            outfit_url = '{}{}'.format('https://www.polyvore.com', outfit_url)
            yield scrapy.Request(outfit_url, self.parse_outfit)


    def parse_outfit(self, response):
        """
        Parses the outfit page and extracts images.
        """
        img_urls = [img.extract() for img in response.xpath('//img[@class="img_size_m"]/@src')[:5]]
        name = response.xpath('//title/text()').extract_first()
        name = name[:-11] if name else None

        # num_views, num_likes = re.findall('hours ago. ([0-9]*) views. ([0-9]*) likes.',response.text)[0]


        for i, img_url in enumerate(img_urls):
            # the_path = '{}_views{}_likes{}_pic{}'.format(name, num_views, num_likes, str(i))
            the_path = '{}_pic{}'.format(name, str(i))
            the_path = the_path.title().replace(' ', '')
            the_path = the_path.translate({ord(c): "" for c in "!@#$%^&*[]{};,./<>?\|`~-=+"})
            the_path += '.jpg'
            yield OutfitspiderItem(outfit_title=name,
                                the_path=the_path,
                                # num_views=num_views,
                                # num_likes=num_likes,
                                item_number=i,
                                image_urls=[img_url])
