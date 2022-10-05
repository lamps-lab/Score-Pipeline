"""
Object models for the Processing Pipeline to generate features for the DARPA SCORE project
-----------------Includes the pre-processing step for the Predition Market----------------
"""

from fuzzywuzzy import process

_author_ = "Arjun Menon"
_copyright_ = "Copyright 2021, Penn State University"
_license_ = ""
_version_ = "1.0"
_maintainer_ = "Sai Ajay"
_email_ = "svm6277@psu.edu"


class Paper:
    """
    A paper represents the programming model that captures meta-features and features associated with
    a scientific paper of interest.
    """

    def __init__(self):
        self.doi = None
        self.ta3_pid = None
        self.title = None
        self.sjr = 0
        self.uni_rank = 0
        self.year = 0
        self.funded = 0
        self.cited_by_count = 0
        self.self_citations = 0
        self.citations = []
        self.authors = []
        self.affiliations = []
        self.ack_pairs = []
        self.subject = ''
        self.subject_code = ''
        self.abstract = ''
        self.influential_references_methodology = 0

    def set_self_citations(self):
        """
        Identify self-citations made in the references section. Count the number of occurrences wherein
        atleast one of the paper authors' name has a fuzzymatch score > 90 with an author listed in a reference.
        Self-citation is computed as a ratio of this number against the total number of references.
        """
        authors = [author.surname for author in self.authors]
        for citation in self.citations:
            citation_authors = [author.surname for author in citation.authors]
            if not citation_authors:
                continue
            for author in authors:
                match = process.extractOne(author, citation_authors)
                if match[1] > 90:
                    self.self_citations += 1
                    break
        if self.self_citations == 0:
            return 0
        else:
            return self.self_citations/len(self.citations)


class Author:
    """
    Simple author programming model
    """
    def __init__(self):
        self.first_name = None
        self.middle_name = None
        self.surname = None
        self.name = None

    def set_name(self):
        """
        Sets full name of an author based on the attribute values
        """
        self.name = ' '.join([self.first_name, self.middle_name, self.surname])


class Organization:
    """
    Simple organization/affiliation entity programming model
    """
    def __init__(self):
        self.type = None
        self.name = None
        self.address = Address()


class Address:
    """
    Simple address model
    """
    def __init__(self):
        self.place = None
        self.region = None
        self.country = None


class Citation:
    """
    Simple citation programming model
    """
    def __init__(self):
        self.title = None
        self.authors = []
        self.source = None
        self.doi = None
        self.publish_year = None
