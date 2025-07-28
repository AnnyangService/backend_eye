#!/usr/bin/env python3
"""
RAG 검색 서비스 테스트 스크립트
"""

import os
import sys
import logging
from app import create_app

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_retrieval_service():
    """RAG 검색 서비스를 테스트합니다."""
    
    print("🔍 RAG 검색 서비스 테스트 시작")
    print("=" * 50)
    
    try:
        # Flask 앱 생성
        app = create_app()
        
        with app.app_context():
            # 서비스 초기화
            print("📦 서비스 초기화 중...")
            from app.chatbot.retrieval import RAGRetrievalService
            retrieval_service = RAGRetrievalService()
            print("✅ 서비스 초기화 완료")
            
            # 테스트 질문들
            test_queries = [
                "각막궤양이란 무엇인가요?",
                "결막염의 증상은 무엇인가요?",
                "안검염의 치료 방법은?",
                "비궤양성 각막염의 원인은?",
                "고양이 눈 질병의 예방법은?"
            ]
            
            print("\n🔎 검색 테스트 시작")
            print("-" * 30)
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n📝 테스트 {i}: {query}")
                print("-" * 40)
                
                try:
                    # 유사한 청크 검색
                    results = retrieval_service.search_similar_chunks(query, top_k=3)
                    
                    if results:
                        print(f"✅ 검색 완료: {len(results)}개 결과")
                        
                        for j, result in enumerate(results, 1):
                            similarity = result['similarity']
                            content = result['chunk']['content']
                            source = result['chunk'].get('source', None)
                            keywords = result['chunk'].get('keywords', None)
                            
                            print(f"\n📄 결과 {j} (유사도: {similarity:.3f})")
                            if source:
                                print(f"문서 ID(source): {source}")
                            print(f"내용: {content[:100]}...")
                            if keywords:
                                print(f"키워드: {keywords}")
                            if len(content) > 100:
                                print(f"      {content[100:200]}...")
                    else:
                        print("❌ 검색 결과가 없습니다.")
                        
                except Exception as e:
                    print(f"❌ 검색 실패: {str(e)}")
            
            print("\n" + "=" * 50)
            print("🎉 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 서비스 초기화 실패: {str(e)}")
        return False
    
    return True

def test_single_query():
    """단일 질문으로 테스트합니다."""
    
    print("🔍 단일 질문 테스트")
    print("=" * 30)
    
    try:
        # Flask 앱 생성
        app = create_app()
        
        with app.app_context():
            # 서비스 초기화
            from app.chatbot.retrieval import RAGRetrievalService
            retrieval_service = RAGRetrievalService()
            
            # 사용자 입력 받기
            query = input("질문을 입력하세요: ").strip()
            
            if not query:
                print("❌ 질문을 입력해주세요.")
                return
            
            print(f"\n🔎 검색 중: {query}")
            print("-" * 30)
            
            # 검색 실행
            results = retrieval_service.search_similar_chunks(query, top_k=5)
            
            if results:
                print(f"✅ 검색 완료: {len(results)}개 결과")
                
                for i, result in enumerate(results, 1):
                    similarity = result['similarity']
                    content = result['chunk']['content']
                    source = result['chunk'].get('source', None)
                    keywords = result['chunk'].get('keywords', None)
                    
                    print(f"\n📄 결과 {i} (유사도: {similarity:.3f})")
                    if source:
                        print(f"문서 ID(source): {source}")
                    print(f"내용: {content}")
                    if keywords:
                        print(f"키워드: {keywords}")
                    print("-" * 30)
            else:
                print("❌ 검색 결과가 없습니다.")
                
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")

if __name__ == "__main__":
    print("RAG 검색 서비스 테스트")
    print("1. 전체 테스트")
    print("2. 단일 질문 테스트")
    
    choice = input("\n선택하세요 (1 또는 2): ").strip()
    
    if choice == "1":
        test_retrieval_service()
    elif choice == "2":
        test_single_query()
    else:
        print("❌ 잘못된 선택입니다.") 