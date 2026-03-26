"""
Chatbot Service for SumerTrip Event Management Website
Handles customer inquiries about trips, events, bookings, and general questions
"""

from typing import Optional
from fal_client import fal_client


# System prompt with all the knowledge about SumerTrip
SUMERTRIP_SYSTEM_PROMPT = """أنت مساعد ذكي لموقع "سومر تريب" (SumerTrip) - منصة عراقية متخصصة بالسياحة والرحلات والفعاليات في العراق.

## معلومات عن الموقع:
- **اسم المنصة**: سومر تريب (SumerTrip)
- **الوصف**: منصة سياحية عراقية تقدم رحلات سياحية وثقافية وفعاليات في مختلف المحافظات العراقية
- **اللغة الرئيسية**: العربية (مع دعم الإنجليزية)

## أنواع الخدمات المتاحة:
1. **الرحلات السياحية (Trips)**:
   - رحلات التراث (Heritage): زيارة المواقع الأثرية مثل زيقورة أور، بابل، نينوى، حاترا
   - رحلات استكشافية (Trips): جولات في المدن مثل بغداد، البصرة، أربيل، النجف
   - رحلات التخييم (Camping): الأهوار، جبال كردستان، وادي شقلاوة

2. **الفعاليات (Events)**:
   - المهرجانات الثقافية والفنية
   - المعارض (معرض الكتاب الدولي)
   - الفعاليات الموسمية

## الوجهات الرئيسية:
- **بغداد**: العاصمة، متحف العراق، شارع المتنبي، شارع الرشيد
- **بابل**: آثار بابل، بوابة عشتار، قصر نبوخذنصر
- **أربيل**: قلعة أربيل (يونسكو)، البازار التاريخي
- **النجف**: مرقد الإمام علي (ع)، وادي السلام
- **البصرة**: شط العرب، سوق العشار، منزل السندباد
- **الأهوار**: موقع تراث عالمي، ركوب المشحوف، بيوت القصب
- **الموصل/نينوى**: آثار نينوى، قصر سنحاريب
- **كردستان**: شلالات كلي علي بك، جبال أمادية، شقلاوة

## معلومات الأسعار (تقريبية):
- رحلات نصف يوم: 35-55 دولار
- رحلات يوم كامل: 60-130 دولار
- رحلات متعددة الأيام: 150-200 دولار
- الفعاليات: مجانية - 60 دولار

## معلومات الحجز:
- يمكن الحجز عبر الموقع الإلكتروني
- الدفع المسبق مطلوب لتأكيد الحجز
- سياسة الإلغاء: إلغاء مجاني قبل 48 ساعة
- يتم توفير مرشدين سياحيين محترفين

## إرشادات للرد:
1. استخدم اللغة العربية الفصحى السهلة
2. كن ودوداً ومساعداً
3. قدم معلومات دقيقة ومفصلة
4. اقترح رحلات مناسبة حسب اهتمامات المستخدم
5. إذا سُئلت عن شيء خارج نطاق السياحة العراقية، وجه المحادثة بلطف
6. شجع على استكشاف التراث العراقي الغني
7. قدم نصائح سفر مفيدة (الطقس، الملابس، أفضل أوقات الزيارة)

## أوقات الزيارة الموصى بها:
- **الخريف والربيع** (مارس-مايو، سبتمبر-نوفمبر): أفضل الأوقات
- **الشتاء**: مناسب لجنوب العراق
- **الصيف**: مناسب لكردستان (الجبال)

أنت هنا لمساعدة الزوار في اكتشاف جمال العراق وتراثه العريق!"""


class ChatbotService:
    """
    Chatbot service for SumerTrip website
    Handles customer inquiries using Fal.ai OpenRouter
    """

    def __init__(self, custom_system_prompt: Optional[str] = None):
        """
        Initialize chatbot with optional custom system prompt

        Args:
            custom_system_prompt: Override the default system prompt
        """
        self.system_prompt = custom_system_prompt or SUMERTRIP_SYSTEM_PROMPT
        self.conversation_history = []

    async def chat(
        self, message: str, include_history: bool = True, temperature: float = 0.7
    ) -> str:
        """
        Send a message and get a response

        Args:
            message: User message
            include_history: Whether to include conversation history
            temperature: Creativity setting

        Returns:
            str: Assistant response
        """
        # Build the prompt with history if needed
        if include_history and self.conversation_history:
            history_text = "\n".join(
                [
                    f"المستخدم: {h['user']}\nالمساعد: {h['assistant']}"
                    for h in self.conversation_history[-5:]  # Last 5 exchanges
                ]
            )
            full_prompt = (
                f"المحادثة السابقة:\n{history_text}\n\nالرسالة الجديدة: {message}"
            )
        else:
            full_prompt = message

        try:
            response = await fal_client.generate(
                prompt=full_prompt,
                system_prompt=self.system_prompt,
                temperature=temperature,
            )

            assistant_message = response.get(
                "output", "عذراً، حدث خطأ. يرجى المحاولة مرة أخرى."
            )

            # Store in history
            self.conversation_history.append(
                {"user": message, "assistant": assistant_message}
            )

            return assistant_message

        except Exception as e:
            return f"عذراً، حدث خطأ في الاتصال: {str(e)}"

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_history(self) -> list:
        """Get conversation history"""
        return self.conversation_history


# Singleton instance for stateless API usage
chatbot = ChatbotService()


async def get_chat_response(
    message: str, conversation_id: Optional[str] = None, temperature: float = 0.7
) -> dict:
    """
    Stateless chat function for API usage

    Args:
        message: User message
        conversation_id: Optional conversation ID (for future session management)
        temperature: Creativity setting

    Returns:
        dict with response and metadata
    """
    response = await fal_client.generate(
        prompt=message, system_prompt=SUMERTRIP_SYSTEM_PROMPT, temperature=temperature
    )

    return {
        "response": response.get("output", ""),
        "conversation_id": conversation_id,
        "usage": response.get("usage", {}),
    }
