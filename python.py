# python.py

import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo cáo Tài Chính 📊")

# --- Khởi tạo Lịch sử Chat ---
# Sử dụng st.session_state để lưu trữ lịch sử cuộc trò chuyện
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Chào bạn! Hãy tải lên báo cáo tài chính để bắt đầu phân tích. Sau đó, bạn có thể hỏi tôi bất kỳ câu hỏi nào về dữ liệu đã phân tích hoặc kiến thức tài chính tổng quát nhé!"}
    ]

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý giá trị 0 cho mẫu số
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini (Chức năng phân tích tổng thể) ---
# Giữ nguyên hàm này cho phân tích 1 lần sau khi upload file
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Khởi tạo biến df_processed và các chỉ số cơ bản để đảm bảo luôn có sẵn, tránh lỗi tham chiếu
df_processed = None
thanh_toan_hien_hanh_N = "N/A"
thanh_toan_hien_hanh_N_1 = "N/A"

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]   
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán, kiểm tra chia cho 0
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else float('inf')
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else float('inf')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if thanh_toan_hien_hanh_N_1 != float('inf') else "Vô hạn",
                    )
                with col2:
                    delta_value = thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1 if thanh_toan_hien_hanh_N != float('inf') and thanh_toan_hien_hanh_N_1 != float('inf') else "N/A"
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if thanh_toan_hien_hanh_N != float('inf') else "Vô hạn",
                        delta=f"{delta_value:.2f}" if isinstance(delta_value, float) else None
                    )
                    
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
            except ZeroDivisionError:
                 st.warning("Không thể tính chỉ số Thanh toán Hiện hành vì Nợ ngắn hạn bằng 0.")
            
            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI Tổng thể)")
            
            # Chuẩn bị dữ liệu để gửi cho AI (dùng dữ liệu thô và chỉ số)
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, float) and thanh_toan_hien_hanh_N_1 != float('inf') else str(thanh_toan_hien_hanh_N_1), 
                    f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) and thanh_toan_hien_hanh_N != float('inf') else str(thanh_toan_hien_hanh_N)
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích Tổng thể"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    # Hiển thị thông tin yêu cầu tải file khi chưa có file
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

# ==============================================================================
# --- CHỨC NĂNG MỚI: KHUNG CHATBOT HỎI ĐÁP (Chức năng 6) ---
# ==============================================================================
st.markdown("---")
st.subheader("6. Hỗ trợ Hỏi đáp Chuyên sâu (Chatbot Gemini)")
st.caption("Chatbot này có khả năng trả lời các câu hỏi về tài chính, cũng như giải thích thêm về dữ liệu bạn đã tải lên.")

# 1. Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Xử lý input từ người dùng
if prompt := st.chat_input("Hỏi Gemini về các chỉ số, tăng trưởng, hoặc kiến thức tài chính..."):
    # Thêm tin nhắn của người dùng vào lịch sử chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Hiển thị tin nhắn của người dùng ngay lập tức
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Gọi Gemini API để nhận phản hồi
    api_key = st.secrets.get("GEMINI_API_KEY")
    
    with st.chat_message("assistant"):
        if not api_key:
             ai_response = "Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng chức năng chat."
             st.markdown(ai_response)
        else:
            try:
                # Chuẩn bị lịch sử chat cho API (chuyển đổi định dạng)
                history_for_api = [
                    {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                    for m in st.session_state.messages
                ]
                
                client = genai.Client(api_key=api_key)
                model_name = 'gemini-2.5-flash'
                
                # System instruction cho chatbot (thêm bối cảnh về dữ liệu đã upload nếu có)
                system_instruction = "Bạn là một trợ lý tài chính thông minh, vui vẻ và am hiểu. Hãy trả lời các câu hỏi của người dùng về tài chính, kinh tế, hoặc dữ liệu đã được tải lên/phân tích trên ứng dụng này. Nếu dữ liệu phân tích có sẵn, hãy sử dụng nó để trả lời chính xác nhất. Luôn duy trì giọng điệu chuyên nghiệp và thân thiện bằng tiếng Việt."
                
                # Thêm dữ liệu đã xử lý vào bối cảnh đầu tiên của chat (chỉ khi có dữ liệu)
                # Kỹ thuật: chèn thông tin tài chính đã phân tích vào tin nhắn đầu tiên của user để Gemini có bối cảnh.
                if df_processed is not None:
                    financial_summary = (
                        "BỐI CẢNH DỮ LIỆU ĐÃ PHÂN TÍCH:\n"
                        "Người dùng đã tải lên và phân tích báo cáo tài chính. Dữ liệu bảng đã xử lý:\n"
                        f"{df_processed.to_markdown(index=False)}\n"
                        f"Chỉ số Thanh toán Hiện hành Năm trước: {thanh_toan_hien_hanh_N_1}\n"
                        f"Chỉ số Thanh toán Hiện hành Năm sau: {thanh_toan_hien_hanh_N}\n\n"
                        "VUI LÒNG DÙNG DỮ LIỆU TRÊN ĐỂ TRẢ LỜI CÂU HỎI TIẾP THEO:\n"
                    )
                    # Chèn bối cảnh vào tin nhắn cuối cùng của user
                    history_for_api[-1]["parts"][0]["text"] = financial_summary + history_for_api[-1]["parts"][0]["text"]
                
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=history_for_api,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_instruction
                    )
                )
                
                ai_response = response.text
                st.markdown(ai_response)

            except APIError as e:
                ai_response = f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
                st.markdown(ai_response)
            except Exception as e:
                ai_response = f"Đã xảy ra lỗi không xác định trong quá trình chat: {e}"
                st.markdown(ai_response)


    # 4. Thêm phản hồi của AI vào lịch sử chat
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
