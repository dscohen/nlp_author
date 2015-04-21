# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.
import scrapy
from scrapy.http import Request
import re
from scrapy.spider import BaseSpider
from scrapy.selector import HtmlXPathSelector

class AmazonSpider(scrapy.Spider):
    name = "authors"
    allowed_domains = ["amazon.com"]
    start_urls = [
            "http://www.amazon.com/review/top-reviewers",
            "http://www.amazon.com/review/top-reviewers/ref=cm_cr_tr_link_2?ie=UTF8&page=2",
            "http://www.amazon.com/review/top-reviewers/ref=cm_cr_tr_link_3?ie=UTF8&page=3"
            ]

    def parse(self, response):
        hxs = HtmlXPathSelector(response)
        links = hxs.select("//a/@href").extract()
        #We stored already crawled links in this list
        crawledLinks = []
        #Pattern to check proper link
        #http://www.amazon.com/review/top-reviewers?page=20
        linkPattern = re.compile("^(?:ftp|http|https)\:\/\/(?:www\.amazon)\.(?:com)\/(?:review)((\?page=[1-4]{1})|(\/(?:top-reviewers)(?:\/ref=cm.*[1-4]{1})))?$")
        for link in links:
            # If it is a proper link and is not checked yet, yield it to the Spider
            if linkPattern.match(link) and not link in crawledLinks:
                crawledLinks.append(link)
                yield Request(link, self.parse)
        titles	= hxs.select('//h1[@class="post_title"]/a/text()').extract()
        for title in titles:
            item 			= AmazonItem()
            item["title"] 	= title
            print title
            yield item
        filename = "top_reviewers"
        with open(filename, 'a') as f:
            f.write(response.body)
