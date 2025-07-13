"""
문서 청킹 모듈
마크다운 문서를 ## 헤더 기준으로 청킹하고 키워드를 추출합니다.
"""

import re
from typing import List, Dict, Any
from .keyword_extractor import extract_keywords_from_text


def chunk_document_by_headers(markdown_text: str, doc_name: str = "") -> List[Dict[str, Any]]:
    """
    마크다운 문서를 ## 헤더 기준으로 청킹합니다.
    
    Args:
        markdown_text (str): 마크다운 문서 텍스트
        doc_name (str): 문서 이름
        
    Returns:
        List[Dict[str, Any]]: 청킹된 문서 리스트
    """
    # ## 헤더로 문서를 분할
    sections = re.split(r'^##\s+', markdown_text, flags=re.MULTILINE)
    
    chunks = []
    
    for i, section in enumerate(sections):
        if not section.strip():
            continue
            
        # 첫 번째 섹션은 제목 섹션 (## 없음)
        if i == 0:
            # 제목 추출
            title_match = re.match(r'^#\s+(.+)$', section, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
                # 제목 다음 내용
                content = re.sub(r'^#\s+.+$', '', section, flags=re.MULTILINE).strip()
                if content:
                    keywords = extract_keywords_from_text(content)
                    chunks.append({
                        "content": content,
                        "keywords": keywords,
                        "id": f"{doc_name}.정의" if doc_name else "정의"
                    })
        else:
            # ## 헤더가 있는 섹션들
            lines = section.split('\n', 1)
            if len(lines) >= 2:
                header = lines[0].strip()
                content = lines[1].strip()
                
                if content:
                    keywords = extract_keywords_from_text(content)
                    
                    # 섹션 이름 매핑
                    section_name = map_section_name(header)
                    
                    chunks.append({
                        "content": content,
                        "keywords": keywords,
                        "id": f"{doc_name}.{section_name}" if doc_name else section_name
                    })
    
    return chunks


def map_section_name(header: str) -> str:
    """
    헤더를 섹션 이름으로 매핑합니다.
    
    Args:
        header (str): 헤더 텍스트
        
    Returns:
        str: 매핑된 섹션 이름
    """
    header_lower = header.lower()
    
    if "증상" in header_lower:
        return "증상"
    elif "검사" in header_lower:
        return "검사"
    elif "치료" in header_lower:
        return "치료"
    elif "예후" in header_lower:
        return "예후"
    elif "정의" in header_lower or "이란" in header_lower:
        return "정의"
    else:
        return "기타"


def process_all_documents(documents_dir: str) -> List[Dict[str, Any]]:
    """
    모든 문서를 처리하여 청킹합니다.
    
    Args:
        documents_dir (str): 문서 디렉토리 경로
        
    Returns:
        List[Dict[str, Any]]: 모든 문서의 청킹 결과
    """
    import os
    import glob
    
    all_chunks = []
    
    # 마크다운 파일들 찾기
    md_files = glob.glob(os.path.join(documents_dir, "*.md"))
    
    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 문서 이름 추출
        doc_name = os.path.basename(md_file).replace('.md', '')
        
        # 문서 청킹
        chunks = chunk_document_by_headers(content, doc_name)
        
        all_chunks.extend(chunks)
    
    return all_chunks 