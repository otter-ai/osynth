"""Calendar scheduling utilities."""
from __future__ import annotations

from datetime import date, datetime, timedelta

from ..models.calendar import CalendarEvent, RecurringMeeting, Schedule


class CalendarScheduler:
    """Handles scheduling logic for the workspace calendar."""

    def __init__(
        self,
        working_hours: tuple[int, int] = (9, 17),
        working_days: list[int] | None = None,
        timezone: str = "America/Los_Angeles",
    ):
        self.working_hours = working_hours
        self.working_days = working_days or [0, 1, 2, 3, 4]
        self.timezone = timezone

    def expand_recurring_meetings(
        self,
        recurring_meetings: list[RecurringMeeting],
        start_date: date,
        end_date: date,
    ) -> list[CalendarEvent]:
        events = []
        for rm in recurring_meetings:
            events.extend(self._expand_single_recurring(rm, start_date, end_date))
        return events

    def _expand_single_recurring(
        self,
        meeting: RecurringMeeting,
        start_date: date,
        end_date: date,
    ) -> list[CalendarEvent]:
        events = []
        current = start_date
        parts = meeting.time_of_day.split(":")
        hour, minute = int(parts[0]), int(parts[1])

        while current <= end_date:
            if self._should_occur(meeting, current):
                start_time = datetime(current.year, current.month, current.day, hour, minute)
                end_time = start_time + timedelta(minutes=meeting.duration_minutes)
                event = CalendarEvent(
                    id=f"cal_{meeting.id}_{current.isoformat()}",
                    title=meeting.title,
                    meeting_type=meeting.meeting_type,
                    start_time=start_time,
                    end_time=end_time,
                    organizer_id=meeting.organizer_id,
                    attendee_ids=list(meeting.attendee_ids),
                    recurring_meeting_id=meeting.id,
                    description=meeting.description,
                    labels=meeting.labels.model_copy(),
                )
                events.append(event)
            current += timedelta(days=1)

        return events

    def _should_occur(self, meeting: RecurringMeeting, d: date) -> bool:
        weekday = d.weekday()
        if weekday not in self.working_days:
            return False
        match meeting.cadence:
            case "daily":
                return True
            case "weekly":
                return meeting.day_of_week is not None and weekday == meeting.day_of_week
            case "biweekly":
                if meeting.day_of_week is None or weekday != meeting.day_of_week:
                    return False
                return d.isocalendar()[1] % 2 == 0
            case "monthly":
                if meeting.day_of_week is not None and weekday != meeting.day_of_week:
                    return False
                return d.day <= 7
            case _:
                return False

    def find_available_slot(
        self,
        attendee_ids: list[str],
        target_date: date,
        duration_minutes: int,
        schedule: Schedule,
        preferred_time: str | None = None,
    ) -> datetime | None:
        if not self.is_working_day(target_date):
            return None

        wh_start, wh_end = self.working_hours

        if preferred_time:
            parts = preferred_time.split(":")
            pref_hour, pref_minute = int(parts[0]), int(parts[1])
            candidate = datetime(target_date.year, target_date.month, target_date.day, pref_hour, pref_minute)
            candidate_end = candidate + timedelta(minutes=duration_minutes)
            if (
                candidate.hour >= wh_start
                and candidate_end.hour <= wh_end
                and not self._has_any_conflict(attendee_ids, candidate, candidate_end, schedule)
            ):
                return candidate

        for hour in range(wh_start, wh_end):
            for minute in [0, 30]:
                candidate = datetime(target_date.year, target_date.month, target_date.day, hour, minute)
                candidate_end = candidate + timedelta(minutes=duration_minutes)
                if candidate_end > datetime(target_date.year, target_date.month, target_date.day, wh_end, 0):
                    continue
                if not self._has_any_conflict(attendee_ids, candidate, candidate_end, schedule):
                    return candidate
        return None

    def find_available_slot_in_week(
        self,
        attendee_ids: list[str],
        week_start: date,
        duration_minutes: int,
        schedule: Schedule,
        preferred_day: int | None = None,
        preferred_time: str | None = None,
    ) -> datetime | None:
        days_to_try: list[date] = []
        if preferred_day is not None:
            target = week_start + timedelta(days=preferred_day - week_start.weekday())
            if week_start <= target <= week_start + timedelta(days=6):
                days_to_try.append(target)
        for i in range(7):
            d = week_start + timedelta(days=i)
            if d.weekday() in self.working_days and d not in days_to_try:
                days_to_try.append(d)
        for d in days_to_try:
            slot = self.find_available_slot(attendee_ids, d, duration_minutes, schedule, preferred_time)
            if slot is not None:
                return slot
        return None

    def _has_any_conflict(
        self, attendee_ids: list[str], start: datetime, end: datetime, schedule: Schedule
    ) -> bool:
        for pid in attendee_ids:
            if schedule.has_conflict(pid, start, end):
                return True
        return False

    def is_working_day(self, d: date) -> bool:
        return d.weekday() in self.working_days

    def clamp_to_working_day(self, target: date) -> date:
        """Return target if it's a working day, else the nearest working day."""
        if self.is_working_day(target):
            return target
        # Search forward and backward up to 7 days
        for offset in range(1, 8):
            for delta in (offset, -offset):
                candidate = target + timedelta(days=delta)
                if self.is_working_day(candidate):
                    return candidate
        return target  # fallback if no working days at all

    def get_working_days_in_range(self, start: date, end: date) -> list[date]:
        days = []
        current = start
        while current <= end:
            if self.is_working_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days
