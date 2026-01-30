from html2text  import html2text
import bs4
from bs4 import BeautifulSoup

def soup_to_markdown(extracted_content) -> str:
    if not extracted_content:
        raise ValueError("No content to convert to markdown")
    extracted_html:str   = str(extracted_content)
    markdown_content:str = html2text(extracted_html, bodywidth=0)
    return markdown_content.strip()


def select_between_start_n_end(soup:BeautifulSoup, start_element_excluded: bs4.element.Tag, end_element_excluded: bs4.element.Tag):
    if start_element_excluded and end_element_excluded:
        # Create a range to select the content between clear_div and content_div
        extracted_content = []
        current_element = start_element_excluded.find_next_sibling()

        while current_element and current_element != end_element_excluded:
            extracted_content.append(current_element)
            current_element = current_element.find_next_sibling()

        # Now, you can work with the extracted content as needed.
        return insert_multiple_elements_inside_div(soup, extracted_content)

def insert_multiple_elements_inside_div(soup:BeautifulSoup, elements: list):
    div = soup.new_tag("div")
    for element in elements:
        div.append(element)
    return div
