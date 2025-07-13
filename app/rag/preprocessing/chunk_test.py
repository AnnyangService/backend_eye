"""
청킹된 데이터 저장 및 불러오기 모듈
"""

import os
import json
from typing import List, Dict, Any
from .document_chunker import process_all_documents


def save_chunks_to_json(chunks: List[Dict[str, Any]], output_dir: str = "chunks") -> str:
    """
    청킹된 데이터를 JSON 파일로 저장합니다.
    
    Args:
        chunks (List[Dict[str, Any]]): 저장할 청킹 데이터
        output_dir (str): 저장할 디렉토리
        
    Returns:
        str: 저장된 파일 경로
    """
    # 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 생성
    output_file = os.path.join(output_dir, "chunks.json")
    
    # JSON으로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"청킹 데이터가 저장되었습니다: {output_file}")
    print(f"총 {len(chunks)}개의 청킹이 저장되었습니다.")
    
    return output_file


def load_chunks_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    JSON 파일에서 청킹된 데이터를 불러옵니다.
    
    Args:
        file_path (str): 불러올 파일 경로
        
    Returns:
        List[Dict[str, Any]]: 불러온 청킹 데이터
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"청킹 데이터를 불러왔습니다: {file_path}")
    print(f"총 {len(chunks)}개의 청킹을 불러왔습니다.")
    
    return chunks


def process_and_save_chunks(documents_dir: str, output_dir: str = "chunks") -> str:
    """
    문서를 청킹하고 저장합니다.
    
    Args:
        documents_dir (str): 문서 디렉토리 경로
        output_dir (str): 저장할 디렉토리
        
    Returns:
        str: 저장된 파일 경로
    """
    print("문서 청킹을 시작합니다...")
    
    # 문서 청킹
    chunks = process_all_documents(documents_dir)
    
    # 저장
    output_file = save_chunks_to_json(chunks, output_dir)
    
    return output_file


def get_chunks_info(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    청킹 데이터의 통계 정보를 반환합니다.
    
    Args:
        chunks (List[Dict[str, Any]]): 청킹 데이터
        
    Returns:
        Dict[str, Any]: 통계 정보
    """
    # 문서별 통계
    doc_stats = {}
    total_keywords = 0
    
    for chunk in chunks:
        # ID에서 문서명 추출
        doc_name = chunk['id'].split('.')[0] if '.' in chunk['id'] else 'unknown'
        if doc_name not in doc_stats:
            doc_stats[doc_name] = {
                'chunk_count': 0,
                'total_keywords': 0,
                'sections': set()
            }
        
        doc_stats[doc_name]['chunk_count'] += 1
        doc_stats[doc_name]['total_keywords'] += len(chunk['keywords'])
        total_keywords += len(chunk['keywords'])
        
        # ID에서 섹션 추출
        section = chunk['id'].split('.')[-1] if '.' in chunk['id'] else 'unknown'
        doc_stats[doc_name]['sections'].add(section)
    
    # 섹션별 통계
    section_stats = {}
    for chunk in chunks:
        section = chunk['id'].split('.')[-1] if '.' in chunk['id'] else 'unknown'
        if section not in section_stats:
            section_stats[section] = 0
        section_stats[section] += 1
    
    return {
        'total_chunks': len(chunks),
        'total_keywords': total_keywords,
        'documents': doc_stats,
        'sections': section_stats
    }


def print_chunks_info(chunks: List[Dict[str, Any]]):
    """
    청킹 데이터의 통계 정보를 출력합니다.
    
    Args:
        chunks (List[Dict[str, Any]]): 청킹 데이터
    """
    info = get_chunks_info(chunks)
    
    print("=== 청킹 데이터 통계 ===")
    print(f"전체 청킹 수: {info['total_chunks']}")
    print(f"전체 키워드 수: {info['total_keywords']}")
    print()
    
    print("문서별 통계:")
    for doc_name, stats in info['documents'].items():
        print(f"  {doc_name}:")
        print(f"    청킹 수: {stats['chunk_count']}")
        print(f"    총 키워드 수: {stats['total_keywords']}")
        print(f"    섹션: {', '.join(stats['sections'])}")
        print()
    
    print("섹션별 통계:")
    for section, count in info['sections'].items():
        print(f"  {section}: {count}개")
    print()


if __name__ == "__main__":
    # 문서 디렉토리 경로
    documents_dir = os.path.join(os.path.dirname(__file__), "..", "documents")
    
    # 청킹 및 저장
    process_and_save_chunks(documents_dir, "chunks") 