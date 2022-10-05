from tamu_features import PaperInfoCrawler, DataProcessor
import pdb

VENUE_METADATA_FILE = r'C:\Users\arjun\repos\score_psu\pipeline\tamu_features\venue_meta\all_venues.csv'
TRAINING_DIR = r'C:\Users\arjun\repos\score_psu\pipeline\tamu_features\training_data\\'
INPUT_FILE = r'C:\Users\arjun\dev\sample3.csv'


if __name__ == "__main__":
    crawler = PaperInfoCrawler(INPUT_FILE, VENUE_METADATA_FILE)
    print("Crawling Info..\n")
    # Get paper id from API in addition to meta file venue,auth,not found
    # venue_df, auth_df, citations_df = crawler.simple_crawl(p_id, issn, auth,citations)
    venue_df, auth_df, downstream_df, citations_df = crawler.crawl('gw38')
    # venue_df, auth_df, downstream_df, citations_df = crawler.simple_crawl(p_id, issn, auth, citations)
    # base_df, auth_df, downstream_df, notFoundList = crawler.crawl('gw38')
    print(" \nCrawling Finished. Processing data..\n")
    google_scholar_data = False
    #if len(notFoundList) > 0:
    #    google_scholar_data = True
    data_processor = DataProcessor(TRAINING_DIR, google_scholar_data)
    # No downstream for now
    processed_df, imputed_list = data_processor.processData(venue_df, auth_df, downstream_df)