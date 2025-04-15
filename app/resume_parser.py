import fitz  # PyMuPDF
import re
import argparse
import json
import os
from datetime import datetime

class ResumeParser:
    def __init__(self, anonymize=True):
        """
        Initialize the ResumeParser
        
        Args:
            anonymize (bool): Whether to anonymize personal information
        """
        self.anonymize = anonymize
    
    def anonymize_text(self, text):
        """
        Anonymize personal and company information
        
        Args:
            text (str): Text to anonymize
            
        Returns:
            str: Anonymized text
        """
        # Anonymize email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Anonymize phone numbers (various formats)
        text = re.sub(r'\b(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', '[PHONE]', text)
        
        # Anonymize URLs
        text = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '[URL]', text)
        text = re.sub(r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '[URL]', text)
        
        # Anonymize addresses (simplistic approach)
        text = re.sub(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Court|Ct|Lane|Ln|Way)\b', '[ADDRESS]', text)
        
        # Anonymize common social media handles
        text = re.sub(r'(?:linkedin\.com|github\.com|twitter\.com)/\S+', '[SOCIAL_MEDIA]', text)
        
        return text

    def parse(self, pdf_path):
        """
        Parse a resume PDF into sections using PyMuPDF
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: Dictionary containing parsed resume sections
        """
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Extract all text
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            
            # Common section headers in resumes
            section_patterns = [
                r'EDUCATION|Education|ACADEMIC BACKGROUND',
                r'EXPERIENCE|Experience|PROFESSIONAL EXPERIENCE|Professional Experience|EMPLOYMENT|Employment',
                r'SKILLS|Skills|TECHNICAL SKILLS|Technical Skills|COMPETENCIES',
                r'PROJECTS|Projects|PROJECT EXPERIENCE|Project Experience|PROJECTS & OUTSIDE EXPERIENCE',
                r'CERTIFICATIONS|Certifications|CERTIFICATES|Certificates',
                r'PUBLICATIONS|Publications',
                r'AWARDS|Awards|HONORS|Honors|ACHIEVEMENTS',
                r'VOLUNTEERING|Volunteering|COMMUNITY SERVICE',
                r'LANGUAGES|Languages',
                r'INTERESTS|Interests|HOBBIES',
                r'REFERENCES|References'
            ]
            
            # Combine patterns for finding section headers
            section_pattern = '|'.join(f'({pattern})' for pattern in section_patterns)
            
            # Find all matches for section headers
            matches = list(re.finditer(f'^({section_pattern})\\s*$', full_text, re.MULTILINE))
            
            if not matches:
                # If no standard sections found, try a more lenient approach
                matches = list(re.finditer(f'({section_pattern})', full_text))
            
            # If still no matches, try to break by lines with all caps
            if not matches:
                matches = list(re.finditer(r'^[A-Z][A-Z\s]+$', full_text, re.MULTILINE))
            
            # Extract sections based on the matches
            sections = {}
            
            # First section is usually the header/personal info
            if matches:
                header_text = full_text[:matches[0].start()].strip()
                sections["HEADER"] = self.anonymize_text(header_text) if self.anonymize else header_text
                sections["FULL_CONTENT"] = self.anonymize_text(full_text) if self.anonymize else full_text
                
                # Extract remaining sections
                for i in range(len(matches)):
                    section_title = matches[i].group(0).strip()
                    
                    # Get section content
                    if i < len(matches) - 1:
                        section_content = full_text[matches[i].end():matches[i+1].start()].strip()
                    else:
                        section_content = full_text[matches[i].end():].strip()
                    
                    # Clean up section title
                    clean_title = re.sub(r'[^a-zA-Z0-9]', '_', section_title).upper()
                    clean_title = re.sub(r'_+', '_', clean_title).strip('_')
                    
                    # Skip empty sections
                    if not section_content:
                        continue
                        
                    # Anonymize if requested
                    if self.anonymize:
                        section_content = self.anonymize_text(section_content)
                    
                    sections[clean_title] = section_content
            else:
                # If no sections detected, just return the whole content
                sections["FULL_CONTENT"] = self.anonymize_text(full_text) if self.anonymize else full_text
                
            return sections
            
        except Exception as e:
            print(f"Error parsing PDF: {e}")
            return {"error": str(e)}
    
    def save_to_json(self, parsed_resume, output_path=None, pdf_path=None):
        """
        Save parsed resume to JSON file
        
        Args:
            parsed_resume (dict): Parsed resume sections
            output_path (str, optional): Output JSON file path
            pdf_path (str, optional): Original PDF path (used to generate output name if output_path not provided)
            
        Returns:
            str: Path to the saved JSON file
        """
        # Determine output path
        if output_path:
            final_output_path = output_path
        elif pdf_path:
            base_name = os.path.basename(pdf_path).replace('.pdf', '')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output_path = f"{base_name}_parsed_{timestamp}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output_path = f"resume_parsed_{timestamp}.json"
        
        # Save to file
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_resume, f, indent=2, ensure_ascii=False)
        
        return final_output_path


def main():
    parser = argparse.ArgumentParser(description="Parse resume PDF into sections")
    parser.add_argument("pdf_path", help="Path to the resume PDF file")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--raw", action="store_true", help="Don't anonymize personal information")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: File not found: {args.pdf_path}")
        return
    
    # Parse the resume
    resume_parser = ResumeParser(anonymize=not args.raw)
    parsed_resume = resume_parser.parse(args.pdf_path)
    
    # Save to file
    output_path = resume_parser.save_to_json(parsed_resume, args.output, args.pdf_path)
    
    print(f"Resume parsed successfully! Output saved to: {output_path}")
    print(f"Found {len(parsed_resume)} sections")
    
    # Print section names
    print("\nSections found:")
    for section in parsed_resume:
        content_preview = parsed_resume[section][:50].replace('\n', ' ')
        print(f"- {section}: {content_preview}...")

if __name__ == "__main__":
    main()